package com.tinkingmaze.hmm.tianchi;

import java.io.IOException;

public class TimeSeriesTest {
	
	private static final int ObservationLength = 120;
	private static final int numberOfObservations = 500;
	private static final int delta = 50;
	private static final String trainFilePath = "D:/MyEclipse/tianchi/target/mars_tianchi_train_data.csv";
	private static final String testFilePath = "D:/MyEclipse/tianchi/target/mars_tianchi_test_data.csv";
	private static final String predictFilePath = "D:/MyEclipse/tianchi/target/mars_tianchi_predict_data.csv";
	public static void main(String[] args) {
		
		TimeSeries test = new TimeSeries(trainFilePath, testFilePath, 
				numberOfObservations, ObservationLength, delta);
		try {
			test.run(predictFilePath);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
