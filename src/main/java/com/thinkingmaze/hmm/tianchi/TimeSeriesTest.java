package com.thinkingmaze.hmm.tianchi;

import java.io.IOException;

public class TimeSeriesTest {
	
	private static final int stuteLength = 50; //9657;
	private static final String trainFilePath = 
			"C:/Users/star/git/PopularMusicPrediction/target/mars_tianchi_train_data.csv";
	private static final String testFilePath = 
			"C:/Users/star/git/PopularMusicPrediction/target/mars_tianchi_test_data.csv";
	private static final String predictFilePath = 
			"C:/Users/star/git/PopularMusicPrediction/target/mars_tianchi_predict_data.csv";
	public static void main(String[] args) {
		
		TimeSeries test;
		try {
			test = new TimeSeries(trainFilePath, testFilePath, predictFilePath, stuteLength);
			test.run();
		} catch (IOException e1) {
			// TODO Auto-generated catch block
			e1.printStackTrace();
		}
	}
}
