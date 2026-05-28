package com.rinha;

import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.module.blackbird.BlackbirdModule;
import com.rinha.dto.FraudRequest;
import com.rinha.dto.FraudResponse;
import com.rinha.service.FraudService;
import io.undertow.Undertow;
import io.undertow.util.Headers;
import io.undertow.util.Methods;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.concurrent.Executor;

public class RinhaApplication {

    // All 6 possible responses (fraudScore = fraudCount/5, fraudCount ∈ 0..5)
    private static final ByteBuffer[] RESPONSES = {
        wrap("{\"approved\":true,\"fraud_score\":0.0}"),
        wrap("{\"approved\":true,\"fraud_score\":0.2}"),
        wrap("{\"approved\":true,\"fraud_score\":0.4}"),
        wrap("{\"approved\":false,\"fraud_score\":0.6}"),
        wrap("{\"approved\":false,\"fraud_score\":0.8}"),
        wrap("{\"approved\":false,\"fraud_score\":1.0}"),
    };

    private static ByteBuffer wrap(String s) {
        return ByteBuffer.wrap(s.getBytes(StandardCharsets.UTF_8)).asReadOnlyBuffer();
    }

    public static void main(String[] args) throws Exception {
        ObjectMapper mapper = new ObjectMapper()
                .configure(DeserializationFeature.FAIL_ON_UNKNOWN_PROPERTIES, false)
                .setPropertyNamingStrategy(PropertyNamingStrategies.SNAKE_CASE)
                .registerModule(new BlackbirdModule());

        FraudService service = new FraudService();
        service.init(mapper);

        // Virtual thread per CPU-bound request task
        Executor vt = task -> Thread.ofVirtual().start(task);

        Undertow server = Undertow.builder()
                .addHttpListener(8080, "0.0.0.0")
                .setIoThreads(2)
                .setWorkerThreads(8)
                .setHandler(exchange -> {
                    String path   = exchange.getRequestPath();
                    var    method = exchange.getRequestMethod();

                    if (Methods.POST.equals(method) && "/fraud-score".equals(path)) {
                        // receiveFullBytes reads body on I/O thread (non-blocking),
                        // then calls the callback with all bytes ready.
                        exchange.getRequestReceiver().receiveFullBytes(
                                (exch, bytes) -> exch.dispatch(vt, () -> {
                                    try {
                                        FraudRequest  req  = mapper.readValue(bytes, FraudRequest.class);
                                        FraudResponse resp = service.evaluate(req);
                                        int idx = Math.max(0, Math.min(5, Math.round(resp.fraudScore() * 5)));
                                        exch.getResponseHeaders().put(Headers.CONTENT_TYPE, "application/json");
                                        exch.setStatusCode(200);
                                        // duplicate() so each send gets its own position/limit
                                        exch.getResponseSender().send(RESPONSES[idx].duplicate());
                                    } catch (Exception e) {
                                        if (!exch.isResponseStarted()) {
                                            exch.setStatusCode(500);
                                        }
                                        exch.endExchange();
                                    }
                                }),
                                (exch, e) -> {
                                    exch.setStatusCode(400);
                                    exch.endExchange();
                                });

                    } else if (Methods.GET.equals(method) && "/ready".equals(path)) {
                        if (service.isReady()) {
                            exchange.setStatusCode(200);
                            exchange.getResponseSender().send("ok");
                        } else {
                            exchange.setStatusCode(503);
                            exchange.getResponseSender().send("loading");
                        }
                    } else {
                        exchange.setStatusCode(404);
                        exchange.endExchange();
                    }
                })
                .build();

        server.start();
        System.out.println("Listening on :8080");
    }
}
