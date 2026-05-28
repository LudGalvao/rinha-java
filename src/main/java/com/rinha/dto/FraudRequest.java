package com.rinha.dto;

import java.util.List;

public record FraudRequest(
        String           id,
        TransactionData  transaction,
        CustomerData     customer,
        MerchantData     merchant,
        TerminalData     terminal,
        LastTransactionData lastTransaction
) {
    public record TransactionData(double amount, int installments, String requestedAt) {}
    public record CustomerData(double avgAmount, int txCount24h, List<String> knownMerchants) {}
    public record MerchantData(String id, String mcc, double avgAmount) {}
    public record TerminalData(boolean isOnline, boolean cardPresent, double kmFromHome) {}
    public record LastTransactionData(String timestamp, double kmFromCurrent) {}
}
