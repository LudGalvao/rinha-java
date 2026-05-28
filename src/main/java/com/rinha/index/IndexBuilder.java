package com.rinha.index;

import com.fasterxml.jackson.core.JsonToken;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.IntStream;
import java.util.zip.GZIPInputStream;

/**
 * Offline tool — run once at Docker build time.
 *
 * Usage:
 *   java -cp rinha-java.jar com.rinha.index.IndexBuilder \
 *        resources/references.json.gz vector_index.bin
 *
 * Algorithm:
 *   1. Stream-parse references.json.gz → float[][] vectors, byte[] labels
 *   2. Run mini-batch k-means (K=1000) on a 50 K sample to get centroids
 *   3. Assign all N vectors to their nearest centroid (parallel)
 *   4. Write binary IVF index (little-endian)
 *
 * Binary format:
 *   int32  K
 *   int32  N
 *   K × 14 × float32  centroids
 *   for each cluster k:
 *     int32  size_k
 *     size_k × 14 × int16   vectors (quantised × 10 000; –1 sentinel → –10 000)
 *     size_k × int8          labels  (0 = legit, 1 = fraud)
 */
public final class IndexBuilder {

    private static final int   DIM          = 14;
    private static final int   K            = 1000;
    private static final int   SAMPLE_SIZE  = 50_000;
    private static final int   MINI_BATCH   = 5_000;
    private static final int   ITERATIONS   = 30;

    public static void main(String[] args) throws Exception {
        if (args.length < 2) {
            System.err.println("Usage: IndexBuilder <references.json.gz> <output.bin>");
            System.exit(1);
        }
        String input  = args[0];
        String output = args[1];

        System.out.println("[1/4] Loading vectors from " + input + " ...");
        long t0 = System.currentTimeMillis();
        VectorSet data = loadVectors(input);
        System.out.printf("      %,d vectors loaded in %.1f s%n",
                data.n, (System.currentTimeMillis() - t0) / 1000.0);

        System.out.println("[2/4] k-means++ init + mini-batch k-means (K=" + K + ") ...");
        t0 = System.currentTimeMillis();
        float[] centroids = buildCentroids(data.vectors, data.n);
        System.out.printf("      Centroids ready in %.1f s%n",
                (System.currentTimeMillis() - t0) / 1000.0);

        System.out.println("[3/4] Assigning vectors to clusters ...");
        t0 = System.currentTimeMillis();
        int[] assignments = assignAll(data.vectors, data.n, centroids);
        System.out.printf("      Assignment done in %.1f s%n",
                (System.currentTimeMillis() - t0) / 1000.0);

        System.out.println("[4/4] Writing binary index to " + output + " ...");
        t0 = System.currentTimeMillis();
        writeBinary(output, centroids, data.vectors, data.labels, assignments, data.n);
        System.out.printf("      Written in %.1f s%n",
                (System.currentTimeMillis() - t0) / 1000.0);

        System.out.println("Done.");
    }

    // -----------------------------------------------------------------------
    // 1 – Load
    // -----------------------------------------------------------------------

    private record VectorSet(float[][] vectors, byte[] labels, int n) {}

    private static VectorSet loadVectors(String path) throws Exception {
        int capacity = 3_100_000;
        float[][] vectors = new float[capacity][DIM];
        byte[]    labels  = new byte[capacity];
        int       n       = 0;

        ObjectMapper mapper = new ObjectMapper();
        try (var gz     = new GZIPInputStream(new BufferedInputStream(new FileInputStream(path), 1 << 17));
             var parser = mapper.createParser(gz)) {

            if (parser.nextToken() != JsonToken.START_ARRAY)
                throw new IllegalStateException("Expected JSON array");

            while (parser.nextToken() == JsonToken.START_OBJECT) {
                float[] vec    = null;
                byte    label  = 0;

                while (parser.nextToken() != JsonToken.END_OBJECT) {
                    String field = parser.getCurrentName();
                    parser.nextToken();
                    if ("vector".equals(field)) {
                        vec = new float[DIM];
                        int d = 0;
                        while (parser.nextToken() != JsonToken.END_ARRAY) {
                            vec[d++] = parser.getFloatValue();
                        }
                    } else if ("label".equals(field)) {
                        label = "fraud".equals(parser.getText()) ? (byte) 1 : (byte) 0;
                    } else {
                        parser.skipChildren();
                    }
                }

                if (vec != null) {
                    if (n == capacity) {
                        capacity = (int) (capacity * 1.5);
                        vectors = Arrays.copyOf(vectors, capacity);
                        labels  = Arrays.copyOf(labels,  capacity);
                    }
                    vectors[n] = vec;
                    labels[n]  = label;
                    n++;
                }
            }
        }
        return new VectorSet(vectors, labels, n);
    }

    // -----------------------------------------------------------------------
    // 2 – Centroids (k-means++ init + mini-batch k-means)
    // -----------------------------------------------------------------------

    private static float[] buildCentroids(float[][] vectors, int n) {
        Random rng     = new Random(42);
        int    sample  = Math.min(SAMPLE_SIZE, n);

        // Shuffle-select sample (Fisher-Yates on indices)
        int[] sampleIdx = new int[sample];
        for (int i = 0; i < sample; i++) sampleIdx[i] = i;
        for (int i = sample - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            int t = sampleIdx[i]; sampleIdx[i] = sampleIdx[j]; sampleIdx[j] = t;
        }

        // k-means++ initialisation on the sample
        float[] centroids = kmeansppInit(vectors, sampleIdx, sample, K, rng);

        // Mini-batch k-means
        float[] counts = new float[K];          // for per-centroid learning rate
        int[]   assign = new int[sample];

        for (int iter = 0; iter < ITERATIONS; iter++) {
            // Pick mini-batch
            int bs = Math.min(MINI_BATCH, sample);
            // Assign mini-batch
            for (int bi = 0; bi < bs; bi++) {
                float[] v = vectors[sampleIdx[bi]];
                int best  = nearestCentroid(v, centroids, K);
                assign[bi] = best;
            }
            // Update centroids with learning rate 1/(1 + count)
            for (int bi = 0; bi < bs; bi++) {
                int    c   = assign[bi];
                float  eta = 1f / ++counts[c];
                float[] v  = vectors[sampleIdx[bi]];
                int    base = c * DIM;
                for (int d = 0; d < DIM; d++) {
                    centroids[base + d] = (1f - eta) * centroids[base + d] + eta * v[d];
                }
            }
        }
        return centroids;
    }

    private static float[] kmeansppInit(float[][] vectors, int[] sampleIdx, int sample, int k, Random rng) {
        float[] centroids = new float[k * DIM];
        float[] minDist2  = new float[sample];
        Arrays.fill(minDist2, Float.MAX_VALUE);

        int first = rng.nextInt(sample);
        System.arraycopy(vectors[sampleIdx[first]], 0, centroids, 0, DIM);

        for (int ci = 1; ci < k; ci++) {
            // Update min-distances for the new centroid ci-1
            int prevBase = (ci - 1) * DIM;
            for (int i = 0; i < sample; i++) {
                float d2 = dist2f(vectors[sampleIdx[i]], centroids, prevBase);
                if (d2 < minDist2[i]) minDist2[i] = d2;
            }
            // Sample proportional to minDist2
            double sum = 0;
            for (float d : minDist2) sum += d;
            double target = rng.nextDouble() * sum;
            double cum = 0;
            int chosen = 0;
            for (int i = 0; i < sample; i++) {
                cum += minDist2[i];
                if (cum >= target) { chosen = i; break; }
            }
            System.arraycopy(vectors[sampleIdx[chosen]], 0, centroids, ci * DIM, DIM);

            if (ci % 100 == 0)
                System.out.println("      k-means++ init: " + ci + "/" + k + " centroids");
        }
        return centroids;
    }

    // -----------------------------------------------------------------------
    // 3 – Assign all vectors
    // -----------------------------------------------------------------------

    private static int[] assignAll(float[][] vectors, int n, float[] centroids) {
        int[] assignments = new int[n];
        // parallel stream runs on the common ForkJoinPool → uses all available CPUs
        IntStream.range(0, n).parallel().forEach(i ->
                assignments[i] = nearestCentroid(vectors[i], centroids, K));
        return assignments;
    }

    // -----------------------------------------------------------------------
    // 4 – Write binary index
    // -----------------------------------------------------------------------

    private static void writeBinary(String path, float[] centroids,
                                    float[][] vectors, byte[] labels,
                                    int[] assignments, int n) throws IOException {
        // Count cluster sizes
        int[] sizes = new int[K];
        for (int i = 0; i < n; i++) sizes[assignments[i]]++;

        // Build per-cluster offset arrays
        int[][] clusterIdx = new int[K][];
        for (int k = 0; k < K; k++) clusterIdx[k] = new int[sizes[k]];
        int[] pos = new int[K];
        for (int i = 0; i < n; i++) {
            int c = assignments[i];
            clusterIdx[c][pos[c]++] = i;
        }

        try (DataOutputStream out = new DataOutputStream(
                new BufferedOutputStream(new FileOutputStream(path), 1 << 20))) {

            writeIntLE(out, K);
            writeIntLE(out, n);

            // Centroids as float32 LE
            ByteBuffer cb = ByteBuffer.allocate(K * DIM * 4).order(ByteOrder.LITTLE_ENDIAN);
            cb.asFloatBuffer().put(centroids);
            out.write(cb.array());

            // Clusters
            for (int k = 0; k < K; k++) {
                int[] idx = clusterIdx[k];
                writeIntLE(out, idx.length);

                // Quantised vectors: float × 100 → int8
                // 0.0 → 0, 1.0 → 100, sentinel –1.0 → –100
                for (int i : idx) {
                    for (int d = 0; d < DIM; d++) {
                        float fv = vectors[i][d];
                        out.writeByte((byte) Math.round(fv * 100f));
                    }
                }

                // Labels
                for (int i : idx) out.writeByte(labels[i]);

                if (k % 100 == 0) System.out.println("      cluster " + k + "/" + K);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Utilities
    // -----------------------------------------------------------------------

    private static int nearestCentroid(float[] v, float[] centroids, int k) {
        float best  = Float.MAX_VALUE;
        int   bestK = 0;
        for (int ci = 0; ci < k; ci++) {
            float d = dist2f(v, centroids, ci * DIM);
            if (d < best) { best = d; bestK = ci; }
        }
        return bestK;
    }

    private static float dist2f(float[] v, float[] centroids, int base) {
        float d0  = v[0]  - centroids[base];
        float d1  = v[1]  - centroids[base + 1];
        float d2  = v[2]  - centroids[base + 2];
        float d3  = v[3]  - centroids[base + 3];
        float d4  = v[4]  - centroids[base + 4];
        float d5  = v[5]  - centroids[base + 5];
        float d6  = v[6]  - centroids[base + 6];
        float d7  = v[7]  - centroids[base + 7];
        float d8  = v[8]  - centroids[base + 8];
        float d9  = v[9]  - centroids[base + 9];
        float d10 = v[10] - centroids[base + 10];
        float d11 = v[11] - centroids[base + 11];
        float d12 = v[12] - centroids[base + 12];
        float d13 = v[13] - centroids[base + 13];
        return d0*d0 + d1*d1 + d2*d2 + d3*d3 + d4*d4 + d5*d5 + d6*d6
             + d7*d7 + d8*d8 + d9*d9 + d10*d10 + d11*d11 + d12*d12 + d13*d13;
    }

    private static void writeIntLE(DataOutputStream out, int v) throws IOException {
        out.writeByte(v & 0xFF);
        out.writeByte((v >>> 8)  & 0xFF);
        out.writeByte((v >>> 16) & 0xFF);
        out.writeByte((v >>> 24) & 0xFF);
    }
}
