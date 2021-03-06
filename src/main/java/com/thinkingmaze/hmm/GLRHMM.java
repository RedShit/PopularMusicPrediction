package com.thinkingmaze.hmm;

import com.thinkingmaze.hmm.version.Version;

public class GLRHMM extends HMM{

	// Used for training
	public GLRHMM(String name, int stateLength, int deltaLength) {
		super.stateLength = stateLength;
		super.deltaLength = deltaLength;
		super.name = name;
		super.versionHMM = new Version(1);
		initialize();
	}

	// Used for Decoding and Evaluation
	public GLRHMM(String name, double[][] stateTransitProbMatrix,
			double[][] emissionProbMatrix, double[] piMatrix) {
		super.stateTransitProbMatrix = stateTransitProbMatrix;
		super.emissionProbMatrix = emissionProbMatrix;
		super.piMatrix = piMatrix;
		super.stateLength = stateTransitProbMatrix.length;
		super.deltaLength = emissionProbMatrix[0].length;
		super.name = name;
		super.versionHMM = new Version(1);
	}
	
	@Override
	public void initialize() {
		
		stateTransitProbMatrix = new double[stateLength][stateLength];
		emissionProbMatrix = new double[stateLength][deltaLength];
		piMatrix = new double[stateLength];
		
		for (int i = 0; i < stateLength; i++) {
			double sum = 0;
			
			for (int j = 0; j < stateLength; j++) {
				sum += stateTransitProbMatrix[i][j] = Math.random();
			}
			
			for (int j = 0; j < stateLength; j++) {
				stateTransitProbMatrix[i][j] = stateTransitProbMatrix[i][j] / sum;
			}
		}
		
		for (int i = 0; i < stateLength; i++) {
			double sum = 0;
			
			for (int j = 0; j < deltaLength; j++) {
				sum += emissionProbMatrix[i][j] = Math.random();
			}
			
			for (int j = 0; j < deltaLength; j++) {
				emissionProbMatrix[i][j] = emissionProbMatrix[i][j] / sum;
			}
		}
		
		double sum = 0;
		
		for (int j = 0; j < stateLength; j++) {
			sum += piMatrix[j] = Math.random();
		}
		
		for (int j = 0; j < stateLength; j++) {
			piMatrix[j] = piMatrix[j] / sum;
		}
	}
}