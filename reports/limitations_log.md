1. Station: Manali, Chennai (CPCB, TN003), the nearest industrial-area station to the CPCL refinery cluster. It is not fence-line data.
2. Period: 2017-10-15 to 2019-05-21 (584 days), the longest near-continuous stretch. A 411-day gap (2016-08 to 2017-09) and shorter gaps elsewhere were not filled. Gaps up to 5 days were interpolated (15 AQI values).
3. PM10, O3, benzene, toluene and xylene are 100% missing at this station, so no hydrocarbon signal is available.
4. CO was stuck at 0.0 for 32 days (2017-12-08 to 2018-01-08) and spiked to 15-40 on 10-13 Oct 2017: CO is not used as an input; the first days (to 14 Oct 2017) were removed.
5. Only 78 of 601 days are "Poor" or worse, so alert evaluation rests on few days. Less than two years of data: the annual cycle cannot be learned reliably.
6. No meteorological data (wind, rain, temperature, humidity) is available, and spikes are largely weather-driven, so one-day-ahead skill from AQI history alone is limited.
7. Test period (88 days, 23 Feb to 21 May 2019) is calm and has only 5 days above AQI 200, so alert metrics are indicative only.
8. ARIMA order chosen on validation MAE among near-tied candidates (49.6 to 50.9); AIC prefers larger models.
9. LSTM hyperparameters (window 14 days, 32 hidden units, learning rate 1e-3) were not tuned. Results are averaged over 5 random seeds; run-to-run test MAE spread is 0.6 (AQI only) and 1.2 (with pollutants).
10. ARIMA was re-fit on train+validation for the test period, while the LSTM was trained on the training period only (validation used for early stopping).
11. The pollutant LSTM was better on validation (winter) and worse on test (spring): a seasonal shift. The model was not re-selected after seeing the test results.
12. Differences between persistence, ARIMA and the LSTM on the test MAE are within about 1 to 3 AQI points on 88 days, so no model is claimed to be better on average error.
13. Block-bootstrap 95% intervals (7-day blocks) for the MAE difference against persistence all include zero (ARIMA -2.6 to +4.7; LSTM AQI-only -4.5 to +2.0; LSTM with pollutants -5.6 to +7.1): no model is established as better on average error.
14. Window length chosen by validation MAE (7: 48.6, 14: 47.8, 21: 48.7); 14 days used.