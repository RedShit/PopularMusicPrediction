package com.thinkingmaze.recurrent.tianchi;


import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;

/** A very simple DataSetIterator for use in the GravesLSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next character in the sequence.<br>
 * This is done by randomly choosing a position in the text file to start the sequence and
 * (optionally) scanning backwards to a new line (to ensure we don't start half way through a word
 * for example).<br>
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */
public class ArtistIterator implements DataSetIterator {
	private static final long serialVersionUID = -7287833919126626356L;
	private int[][] dataFile;
	private double[][] characters;
	private double[][] realFile;
	private int exampleLength;
	private int miniBatchSize;
	private int numExamplesToFetch;
	private int examplesSoFar = 0;
	private Random rng;
	private int numCharacters;
	private final boolean alwaysRandomCreate;
	private final double lowerBound = 1.2;
	private final double upperBound = 3.0;
	private final int featrueDays = 30;
	
	public ArtistIterator(String path, int miniBatchSize, int exampleSize, int numExamplesToFetch, boolean alwaysRandomCreate ) throws IOException {
		this(path,Charset.defaultCharset(),miniBatchSize,exampleSize,numExamplesToFetch, 10, new Random(),alwaysRandomCreate);
	}

	/**
	 * @param textFilePath Path to text file to use for generating samples
	 * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
	 * @param miniBatchSize Number of examples per mini-batch
	 * @param exampleLength Number of characters in each input/output vector
	 * @param numExamplesToFetch Total number of examples to fetch (must be multiple of miniBatchSize). Used in hasNext() etc methods
	 * @param rng Random number generator, for repeatability if required
	 * @param alwaysRandomCreate if true, we find a new data created by random
	 * @throws IOException If text file cannot  be loaded
	 */
	public ArtistIterator(String textFilePath, Charset textFileEncoding, int miniBatchSize, int exampleLength,
			int numExamplesToFetch, int numCharacters, Random rng, boolean alwaysRandomCreate ) throws IOException {
		if( !new File(textFilePath).exists()) throw new IOException("Could not access file (does not exist): " + textFilePath);
		if(numExamplesToFetch % miniBatchSize != 0 ) throw new IllegalArgumentException("numExamplesToFetch must be a multiple of miniBatchSize");
		if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
		this.exampleLength = exampleLength;
		this.miniBatchSize = miniBatchSize;
		this.numExamplesToFetch = numExamplesToFetch;
		this.numCharacters = numCharacters;
		this.rng = rng;
		this.alwaysRandomCreate = alwaysRandomCreate;
		
		
		//Load file and convert contents to a char[] 
		List<String> lines = Files.readAllLines(new File(textFilePath).toPath(),textFileEncoding);
		int maxSize = lines.size();
		this.characters = new double[maxSize][];
		this.dataFile = new int[maxSize][];
		this.realFile = new double[maxSize][];
		int currIdx = 0;
		for( String line : lines ){
			String[] strs = line.split(",");
			if(strs.length < exampleLength)
				throw new IllegalArgumentException("Data must be more than exampleLength");
			this.dataFile[currIdx] = new int[exampleLength];
			this.realFile[currIdx] = new double[exampleLength+1];
			this.characters[currIdx] = new double[numCharacters];
			this.realFile[currIdx][0] = Double.valueOf(strs[0]);
			for(int i = 0; i < exampleLength; i++){
				this.realFile[currIdx][i+1] = Double.valueOf(strs[i+1]);
			}
			this.characters[currIdx] = getCharacters(this.realFile[currIdx]);
			for(int i = 0; i < exampleLength; i++){
				this.dataFile[currIdx][i] = getCharacterIdx(Double.valueOf(strs[i+1]), this.characters[currIdx]);
			}
			currIdx += 1;
		}
		
		System.out.println("Loaded and converted file: " + maxSize);
	}
	
	private int getCharacterIdx(final double value, final double[] characters){
		// TODO Auto-generated method stub
		double prePredict = 0;
		for(int i = 0; i < this.numCharacters; i++){
			double currPredict = characters[i];
			if(Math.abs(currPredict-value) > Math.abs(prePredict-value)){
				return (i==0)?i:i-1;
			}
			prePredict = currPredict;
		}
		return this.numCharacters-1;
	}
	
	private double[] getCharacters(final double[] realFile) {
		// TODO Auto-generated method stub
		double lValue = Double.MAX_VALUE;
		double rValue = realFile[0];
		for(int i = 1; i <= this.featrueDays; i++){
			lValue = Math.min(lValue, realFile[i]);
		}
		lValue = lValue*lowerBound;
		rValue = rValue*upperBound;
		lValue = Math.min(rValue, lValue);
		double deta = (rValue-lValue)/(numCharacters-1);
		List<Double> res = new ArrayList<Double>();
		for(int i = 0; i < this.numCharacters; i++){
			res.add(lValue+i*deta);
		}
		double[] out = new double[res.size()];
		for(int i = 0; i < res.size(); i++)
			out[i] = res.get(i);
		return out;
	}
	
	public boolean hasNext() {
		if(!this.alwaysRandomCreate)
			return examplesSoFar + miniBatchSize <= this.dataFile.length;
		return examplesSoFar + miniBatchSize <= numExamplesToFetch;
	}
	
	public double[] realData(){
		if(examplesSoFar<=0 || examplesSoFar>numExamplesToFetch)  throw new NoSuchElementException();
		return this.realFile[examplesSoFar-1];
	}
	
	public double characterIdx2Value(int idx){
		if(examplesSoFar<=0 || examplesSoFar>numExamplesToFetch)  throw new NoSuchElementException();
		return this.characters[examplesSoFar-1][idx];
	}
	
	public DataSet next() {
		return next(miniBatchSize);
	}

	public DataSet next(int num) {
		if(!this.alwaysRandomCreate && examplesSoFar+num>this.dataFile.length){
			throw new NoSuchElementException();
		}
		if( examplesSoFar+num > numExamplesToFetch ) throw new NoSuchElementException();
		//Allocate space:
		INDArray input = Nd4j.zeros(num,numCharacters,exampleLength);
		INDArray labels = Nd4j.zeros(num,numCharacters,exampleLength);
				
		//Randomly select a subset of the file. No attempt is made to avoid overlapping subsets
		// of the file in the same minibatch
		for( int i=0; i<num; i++ ){
			int artistIdx = (int) (rng.nextDouble()*this.dataFile.length);
			if(this.alwaysRandomCreate == false)
				artistIdx = examplesSoFar+i;
			int currCharIdx = 0;
			for( int j=0, c=0; j<exampleLength; j++, c++ ){
				int nextCharIdx = this.dataFile[artistIdx][j];              		//Next character to predict
				input.putScalar(new int[]{i,currCharIdx,c}, 1.0);
				double[] label = getLabel(this.characters[artistIdx], this.realFile[artistIdx][j+1]);
				for(int k = 0; k < label.length; k++){
					labels.putScalar(new int[]{i,k,c}, label[k]);
				}
				currCharIdx = nextCharIdx;
			}
		}
		
		examplesSoFar += num;
		return new DataSet(input,labels);
	}

	private double[] getLabel(double[] ds, double real) {
		// TODO Auto-generated method stub
		double Z = 0;
		double[] label = new double[ds.length];
		for(int i = 0; i < ds.length; i++){
			real = Math.max(real, 1.0);
			double e = Math.abs(ds[i]-real)/real;
			label[i] = Math.exp(-e);
			Z += label[i];
		}
		for(int i = 0; i < ds.length; i++){
			label[i] = label[i]/Z;
		}
		return label;
	}

	public int totalExamples() {
		return numExamplesToFetch;
	}

	public int inputColumns() {
		return numCharacters;
	}

	public int totalOutcomes() {
		return numCharacters;
	}
	
	public int getInitialCharacter(){
		return 0;
	}

	public void reset() {
		examplesSoFar = 0;
	}

	public int batch() {
		return miniBatchSize;
	}

	public int cursor() {
		return examplesSoFar;
	}

	public int numExamples() {
		return numExamplesToFetch;
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

}