package com.rinha.service;

import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.rinha.dto.FraudRequest;
import com.rinha.dto.FraudResponse;
import com.rinha.vector.IvfIndex;
import com.rinha.vector.Vectorizer;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

public class FraudService {

    private static final int PROBES       = 5;
    private static final int WARMUP_ITERS = 15_000;

    private final AtomicBoolean ready = new AtomicBoolean(false);
    private IvfIndex   index;
    private Vectorizer vectorizer;

    public void init(ObjectMapper mapper) throws Exception {
        Path base      = Paths.get(System.getProperty("user.dir"));
        Path indexPath = resolve(base, "vector_index.bin");
        Path mccPath   = resolve(base, "mcc_risk.json");

        Map<String, Float> mccRisk = mapper.readValue(
                mccPath.toFile(), new TypeReference<>() {});
        vectorizer = new Vectorizer(mccRisk);

        System.out.println("Loading IVF index (probes=" + PROBES + ")...");
        index = IvfIndex.load(indexPath, PROBES);

        System.out.println("JIT warmup (" + WARMUP_ITERS + " iters)...");
        float[] warm = new float[14];
        Arrays.fill(warm, 0.5f);
        for (int i = 0; i < WARMUP_ITERS; i++) index.search(warm);

        ready.set(true);
        System.out.println("Ready.");
    }

    public boolean isReady() { return ready.get(); }

    public FraudResponse evaluate(FraudRequest req) {
        float[]        vec = vectorizer.vectorize(req);
        IvfIndex.Result res = index.search(vec);
        return new FraudResponse(res.approved(), res.fraudScore());
    }

    private static Path resolve(Path base, String name) {
        Path p = base.resolve(name);
        return p.toFile().exists() ? p : base.getParent().resolve(name);
    }
}
