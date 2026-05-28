# ─────────────────────────────────────────────────────────
# Stage 1 – compile + generate IVF binary index
# ─────────────────────────────────────────────────────────
FROM maven:3.9-eclipse-temurin-21 AS builder
WORKDIR /build

COPY pom.xml .
RUN mvn -q dependency:go-offline

COPY src ./src
RUN mvn -q package -DskipTests

# Copy reference files alongside the JAR
COPY resources/references.json.gz references.json.gz
COPY resources/mcc_risk.json      mcc_risk.json
COPY resources/normalization.json normalization.json

# Shade plugin produces a flat JAR — run IndexBuilder directly with -cp
RUN java -Xmx1g \
    -cp target/rinha-java-1.0.0.jar \
    com.rinha.index.IndexBuilder \
    references.json.gz vector_index.bin

# ─────────────────────────────────────────────────────────
# Stage 2 – lean runtime image
# ─────────────────────────────────────────────────────────
FROM eclipse-temurin:21-jre-alpine
WORKDIR /app

COPY --from=builder /build/target/rinha-java-1.0.0.jar app.jar
COPY --from=builder /build/vector_index.bin             vector_index.bin
COPY --from=builder /build/mcc_risk.json                mcc_risk.json

EXPOSE 8080

# Without Spring Boot the JVM footprint is much smaller:
#   heap: 45 MB index + small buffers → Xmx 80 m is plenty
#   metaspace: ~15–20 MB (no Spring auto-config classes) → cap at 32 m
#   code cache: 16 m (small app, JIT compiles less code)
ENV JAVA_OPTS="\
  -Xmx80m \
  -Xms40m \
  -XX:+UseG1GC \
  -XX:MaxGCPauseMillis=5 \
  -XX:G1HeapRegionSize=1m \
  -XX:+DisableExplicitGC \
  -XX:MaxMetaspaceSize=32m \
  -XX:ReservedCodeCacheSize=16m \
  -Djava.security.egd=file:/dev/./urandom \
"

ENTRYPOINT ["sh", "-c", "exec java $JAVA_OPTS -jar app.jar"]
