package com.rinha.vector;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;

/**
 * IVF (Inverted File Index) for approximate k-nearest-neighbour search.
 *
 * Vectors on disk are stored as int8 (byte) scaled × 100:
 *   0.0 → 0, 1.0 → 100, sentinel –1.0 → –100
 *
 * Using byte instead of short halves the index footprint (42 MB vs 84 MB),
 * which dramatically reduces GC pressure when running inside a 165 MB container.
 * Distance arithmetic stays in int32 space (max per-dim diff² = 200² = 40 000;
 * 14 dims → max sum 560 000, fits in int).
 *
 * Binary file format (little-endian):
 *   int32  K          — number of clusters
 *   int32  N          — total vector count
 *   K × 14 × float32  centroids
 *   for each cluster k:
 *     int32  size_k
 *     size_k × 14 × int8   quantised vectors (row-major, signed byte)
 *     size_k × int8          labels  (0 = legit, 1 = fraud)
 */
public final class IvfIndex {

    private static final int DIM    = 14;
    private static final int KNN    = 5;
    private static final float THRESHOLD = 0.6f;

    private final int numClusters;
    private final int probes;
    private final float[] centroids;     // numClusters × DIM
    private final byte[][] clusterVecs;  // clusterVecs[k] = flat byte[], length = size_k × DIM
    private final byte[][] clusterLabels;

    private IvfIndex(int numClusters, int probes, float[] centroids,
                     byte[][] clusterVecs, byte[][] clusterLabels) {
        this.numClusters   = numClusters;
        this.probes        = probes;
        this.centroids     = centroids;
        this.clusterVecs   = clusterVecs;
        this.clusterLabels = clusterLabels;
    }

    // -----------------------------------------------------------------------
    // Loading
    // -----------------------------------------------------------------------

    public static IvfIndex load(Path path, int probes) throws IOException {
        try (DataInputStream in = new DataInputStream(
                new BufferedInputStream(new FileInputStream(path.toFile()), 1 << 20))) {

            int K = readIntLE(in);
            int N = readIntLE(in);

            // Read centroids
            byte[] cRaw = in.readNBytes(K * DIM * 4);
            float[] centroids = new float[K * DIM];
            ByteBuffer.wrap(cRaw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(centroids);

            // Read cluster data (byte-quantised vectors)
            byte[][] clusterVecs   = new byte[K][];
            byte[][] clusterLabels = new byte[K][];

            for (int k = 0; k < K; k++) {
                int size = readIntLE(in);
                clusterVecs[k]   = in.readNBytes(size * DIM); // 1 byte per float
                clusterLabels[k] = in.readNBytes(size);
            }

            return new IvfIndex(K, probes, centroids, clusterVecs, clusterLabels);
        }
    }

    // -----------------------------------------------------------------------
    // Query
    // -----------------------------------------------------------------------

    public record Result(boolean approved, float fraudScore) {}

    public Result search(float[] query) {
        // 1 – find nearest `probes` centroids
        int[] nearest = nearestCentroids(query);

        // 2 – search those clusters; maintain a max-heap of size KNN
        //     distances in int32 space (byte scale: max dim diff = 200, diff² = 40000; 14 dims max = 560000)
        int[]  heapDist  = new int[KNN];
        byte[] heapLabel = new byte[KNN];
        int    filled    = 0;
        int    heapMax   = Integer.MAX_VALUE;

        // Pre-scale query to int space once (× 100)
        int[] qInt = new int[DIM];
        for (int d = 0; d < DIM; d++) {
            qInt[d] = Math.round(query[d] * 100f);
        }

        for (int ci : nearest) {
            byte[] vecs   = clusterVecs[ci];
            byte[] labels = clusterLabels[ci];
            int    n      = vecs.length / DIM;

            for (int i = 0; i < n; i++) {
                int dist2 = dist2Int(qInt, vecs, i * DIM);

                if (filled < KNN) {
                    heapDist[filled]  = dist2;
                    heapLabel[filled] = labels[i];
                    filled++;
                    if (filled == KNN) {
                        buildMaxHeap(heapDist, heapLabel, KNN);
                        heapMax = heapDist[0];
                    }
                } else if (dist2 < heapMax) {
                    heapDist[0]  = dist2;
                    heapLabel[0] = labels[i];
                    siftDown(heapDist, heapLabel, 0, KNN);
                    heapMax = heapDist[0];
                }
            }
        }

        // 3 – count fraud labels in the top-5
        int fraudCount = 0;
        for (int i = 0; i < KNN; i++) {
            if (heapLabel[i] == 1) fraudCount++;
        }

        float score = fraudCount / (float) KNN;
        return new Result(score < THRESHOLD, score);
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    private int[] nearestCentroids(float[] query) {
        // Single pass with a max-heap of size `probes`.
        // Avoids allocating float[K] + boolean[K] (5 KB) on every call.
        float[] heapDist = new float[probes];
        int[]   heapIdx  = new int[probes];
        java.util.Arrays.fill(heapDist, Float.MAX_VALUE);
        int filled = 0;

        for (int k = 0; k < numClusters; k++) {
            float d = 0;
            int base = k * DIM;
            for (int dim = 0; dim < DIM; dim++) {
                float diff = query[dim] - centroids[base + dim];
                d += diff * diff;
            }
            if (filled < probes) {
                heapDist[filled] = d;
                heapIdx[filled]  = k;
                filled++;
                if (filled == probes) buildMaxHeapFloat(heapDist, heapIdx, probes);
            } else if (d < heapDist[0]) {
                heapDist[0] = d;
                heapIdx[0]  = k;
                siftDownFloat(heapDist, heapIdx, 0, probes);
            }
        }
        return heapIdx;
    }

    private static void buildMaxHeapFloat(float[] dist, int[] idx, int n) {
        for (int i = n / 2 - 1; i >= 0; i--) siftDownFloat(dist, idx, i, n);
    }

    private static void siftDownFloat(float[] dist, int[] idx, int i, int n) {
        while (true) {
            int left = 2 * i + 1, right = 2 * i + 2, largest = i;
            if (left  < n && dist[left]  > dist[largest]) largest = left;
            if (right < n && dist[right] > dist[largest]) largest = right;
            if (largest == i) break;
            float td = dist[i];  dist[i]  = dist[largest];  dist[largest]  = td;
            int   ti = idx[i];   idx[i]   = idx[largest];   idx[largest]   = ti;
            i = largest;
        }
    }

    private static int dist2Int(int[] q, byte[] vecs, int base) {
        // Manually unrolled for 14 dims — helps JIT produce vectorised code
        // byte is sign-extended to int automatically; sentinel -100 stays -100
        int d0  = q[0]  - vecs[base];
        int d1  = q[1]  - vecs[base + 1];
        int d2  = q[2]  - vecs[base + 2];
        int d3  = q[3]  - vecs[base + 3];
        int d4  = q[4]  - vecs[base + 4];
        int d5  = q[5]  - vecs[base + 5];
        int d6  = q[6]  - vecs[base + 6];
        int d7  = q[7]  - vecs[base + 7];
        int d8  = q[8]  - vecs[base + 8];
        int d9  = q[9]  - vecs[base + 9];
        int d10 = q[10] - vecs[base + 10];
        int d11 = q[11] - vecs[base + 11];
        int d12 = q[12] - vecs[base + 12];
        int d13 = q[13] - vecs[base + 13];
        return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6
             + d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13;
    }

    // Max-heap (largest distance at index 0) over parallel arrays
    private static void buildMaxHeap(int[] dist, byte[] label, int n) {
        for (int i = n / 2 - 1; i >= 0; i--) siftDown(dist, label, i, n);
    }

    private static void siftDown(int[] dist, byte[] label, int i, int n) {
        while (true) {
            int left = 2 * i + 1, right = 2 * i + 2, largest = i;
            if (left  < n && dist[left]  > dist[largest]) largest = left;
            if (right < n && dist[right] > dist[largest]) largest = right;
            if (largest == i) break;
            int  td = dist[i];  dist[i]  = dist[largest];  dist[largest]  = td;
            byte tl = label[i]; label[i] = label[largest]; label[largest] = tl;
            i = largest;
        }
    }

    // DataInputStream uses big-endian by default; our binary file is little-endian
    private static int readIntLE(DataInputStream in) throws IOException {
        int b0 = in.readUnsignedByte();
        int b1 = in.readUnsignedByte();
        int b2 = in.readUnsignedByte();
        int b3 = in.readUnsignedByte();
        return b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
}
