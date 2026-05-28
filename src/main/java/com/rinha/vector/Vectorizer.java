package com.rinha.vector;

import com.rinha.dto.FraudRequest;

import java.time.ZoneOffset;
import java.time.ZonedDateTime;
import java.time.temporal.ChronoUnit;
import java.util.List;
import java.util.Map;

public class Vectorizer {

    private static final float MAX_AMOUNT               = 10_000f;
    private static final float MAX_INSTALLMENTS         = 12f;
    private static final float AMOUNT_VS_AVG_RATIO      = 10f;
    private static final float MAX_MINUTES              = 1_440f;
    private static final float MAX_KM                   = 1_000f;
    private static final float MAX_TX_COUNT_24H         = 20f;
    private static final float MAX_MERCHANT_AVG_AMOUNT  = 10_000f;

    private final Map<String, Float> mccRisk;

    public Vectorizer(Map<String, Float> mccRisk) {
        this.mccRisk = mccRisk;
    }

    public float[] vectorize(FraudRequest req) {
        float[] v = new float[14];

        FraudRequest.TransactionData tx   = req.transaction();
        FraudRequest.CustomerData    cust = req.customer();
        FraudRequest.MerchantData    merc = req.merchant();
        FraudRequest.TerminalData    term = req.terminal();

        // 0 — amount
        v[0] = clamp((float) tx.amount() / MAX_AMOUNT);

        // 1 — installments
        v[1] = clamp(tx.installments() / MAX_INSTALLMENTS);

        // 2 — amount_vs_avg
        double avg = cust.avgAmount();
        v[2] = (avg <= 0) ? 1f : clamp((float) ((tx.amount() / avg) / AMOUNT_VS_AVG_RATIO));

        // Parse once — used for dimensions 3, 4, and optionally 5
        ZonedDateTime txTime = ZonedDateTime.parse(tx.requestedAt()).withZoneSameInstant(ZoneOffset.UTC);

        // 3 — hour_of_day (0–23 UTC / 23)
        v[3] = txTime.getHour() / 23f;

        // 4 — day_of_week (Mon=0 … Sun=6)
        // DayOfWeek.getValue() → Mon=1, Sun=7
        v[4] = (txTime.getDayOfWeek().getValue() - 1) / 6f;

        // 5 & 6 — last_transaction fields
        FraudRequest.LastTransactionData last = req.lastTransaction();
        if (last == null) {
            v[5] = -1f;
            v[6] = -1f;
        } else {
            ZonedDateTime lastTime = ZonedDateTime.parse(last.timestamp()).withZoneSameInstant(ZoneOffset.UTC);
            long minutes = ChronoUnit.MINUTES.between(lastTime, txTime);
            v[5] = clamp((float) minutes / MAX_MINUTES);
            v[6] = clamp((float) last.kmFromCurrent() / MAX_KM);
        }

        // 7 — km_from_home
        v[7] = clamp((float) term.kmFromHome() / MAX_KM);

        // 8 — tx_count_24h
        v[8] = clamp(cust.txCount24h() / MAX_TX_COUNT_24H);

        // 9 — is_online
        v[9] = term.isOnline() ? 1f : 0f;

        // 10 — card_present
        v[10] = term.cardPresent() ? 1f : 0f;

        // 11 — unknown_merchant (1 = NOT known)
        List<String> known = cust.knownMerchants();
        v[11] = known != null && known.contains(merc.id()) ? 0f : 1f;

        // 12 — mcc_risk (default 0.5 when MCC not in table)
        v[12] = mccRisk.getOrDefault(merc.mcc(), 0.5f);

        // 13 — merchant_avg_amount
        v[13] = clamp((float) merc.avgAmount() / MAX_MERCHANT_AVG_AMOUNT);

        return v;
    }

    private static float clamp(float x) {
        return Math.min(1f, Math.max(0f, x));
    }
}
