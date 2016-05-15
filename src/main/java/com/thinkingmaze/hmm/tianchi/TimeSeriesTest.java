package com.thinkingmaze.hmm.tianchi;

import java.io.IOException;

public class TimeSeriesTest {
	
	private static final int ObservationLength = 120;
	private static final int numberOfObservations = 9657;
	private static final String possionFilePath = 
			"C:/Users/13976/git/PopularMusicPrediction/target/mars_tianchi_poisson_data.csv";
	private static final String trainFilePath = 
			"C:/Users/13976/git/PopularMusicPrediction/target/mars_tianchi_train_data.csv";
	private static final String testFilePath = 
			"C:/Users/13976/git/PopularMusicPrediction/target/mars_tianchi_test_data.csv";
	private static final String predictFilePath = 
			"C:/Users/13976/git/PopularMusicPrediction/target/mars_tianchi_predict_data.csv";
	public static void main(String[] args) {
		
		TimeSeries test;
		try {
			test = new TimeSeries(possionFilePath, trainFilePath, testFilePath, predictFilePath, 
					numberOfObservations, ObservationLength);
			test.run();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}
}
