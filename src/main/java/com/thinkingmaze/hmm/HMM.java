package com.thinkingmaze.hmm;

import com.thinkingmaze.hmm.version.Version;

public abstract class HMM {
	
	Version versionHMM;
	double[][] stateTransitProbMatrix;
	double[][] emissionProbMatrix;
	double[][] piMatrix;
	
	int stateLength;
	int deltaLength;
	int obsListNumber;
	public String name;
	
	public void initialize() {
		versionHMM = new Version(1);
		stateTransitProbMatrix = new double[stateLength][stateLength];
		emissionProbMatrix = new double[stateLength][deltaLength];
		piMatrix = new double[obsListNumber][stateLength];
		
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
		
		for (int i = 0; i < obsListNumber; i++) {
			double sum = 0;
			
			for (int j = 0; j < stateLength; j++) {
				sum += piMatrix[i][j] = Math.random();
			}
			
			for (int j = 0; j < stateLength; j++) {
				piMatrix[i][j] = piMatrix[i][j] / sum;
			}
		}
		
	}
	
	public double getStateTransitProbMatrix(int stateIndex1, int stateIndex2) {
		return stateTransitProbMatrix[stateIndex1 - 1][stateIndex2 - 1];
	}
	
	public double getEmissionProbMatrix(int stateIndex, int obsIndex) {
		return emissionProbMatrix[stateIndex - 1][obsIndex - 1];
	}
	
	public double getPiMatrix(int obsNumber, int stateIndex) {
		return piMatrix[obsNumber - 1][stateIndex - 1];
	}
	
	public double[] getPiMatrix(int obsNumber) {
		return piMatrix[obsNumber - 1];
	}
	
	public int getStateLength() {
		return stateLength;
	}
	
	public int getDeltaLength() {
		return deltaLength;
	}
	
	public void setStateTransitProbMatrix(double transitionProb, int i, int j) {
		this.stateTransitProbMatrix[i - 1][j - 1] = transitionProb ;
		getVersionHMM().update();
	}
	
	public void setStateTransitProbMatrix(double[][] stateTransitProbMatrix) {
		this.stateTransitProbMatrix = stateTransitProbMatrix;
		getVersionHMM().update();
	}
	
	public void setEmissionProbMatrix(double emissionProb, int stateIndex, int vectorIndex) {
		this.emissionProbMatrix[stateIndex - 1][vectorIndex - 1] = emissionProb;
		getVersionHMM().update();
	}
	
	public void setEmissionProbMatrix(double[][] emissionProbMatrix) {
		this.emissionProbMatrix = emissionProbMatrix;
		getVersionHMM().update();
	}
	
	public void setPiMatrix(double pi, int i, int j) {
		this.piMatrix[i - 1][j - 1] = pi;
		getVersionHMM().update();
	}
	
	public void setPiMatrix(double[][] piMatrix) {
		this.piMatrix = piMatrix;
		getVersionHMM().update();
	}

	public String getName() {
		// TODO Auto-generated method stub
		return this.name;
	}
	public Version getVersionHMM(){
		return this.versionHMM;
	}
}
