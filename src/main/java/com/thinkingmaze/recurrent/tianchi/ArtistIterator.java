package com.thinkingmaze.recurrent.tianchi;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.Random;
import java.util.Scanner;

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
	private int[] characters;
	private List<List<Integer>> file;
	private List<List<Integer>> realFile;
	private List<String> artistId;
	private List<Double> actionNumber;
	private int[][] realOutput;
	private int exampleLength;
	private int miniBatchSize;
	private int numExamplesToFetch;
	private int examplesSoFar = 0;
	private Random rng;
	private int numCharacters;
	private final boolean alwaysRandomData;
	private List<String> nextArtistId;
	
	public ArtistIterator(String path, int miniBatchSize, int numExamplesToFetch,
			int exampleLength,boolean alwaysRandomData, String validArtistId ) throws IOException {
		this(path,miniBatchSize,numExamplesToFetch,exampleLength,
				getDefultCharacters(7000),new Random(),alwaysRandomData, validArtistId);
	}

	/**
	 * @param textFilePath Path to text file to use for generating samples
	 * @param textFileEncoding Encoding of the text file. Can try Charset.defaultCharset()
	 * @param miniBatchSize Number of examples per mini-batch
	 * @param exampleLength Number of characters in each input/output vector
	 * @param numExamplesToFetch Total number of examples to fetch (must be multiple of miniBatchSize). Used in hasNext() etc methods
	 * @param rng Random number generator, for repeatability if required
	 * @param alwaysRandomData if true, we find a new data created by random
	 * @throws IOException If text file cannot  be loaded
	 */
	public ArtistIterator(String textFilePath, int miniBatchSize, int numExamplesToFetch, int exampleLength,
			int[] characters, Random rng, boolean alwaysRandomData, String validArtistId ) throws IOException {
		if( !new File(textFilePath).exists()) 
			throw new IOException("Could not access file (does not exist): " + textFilePath);
		if(numExamplesToFetch % miniBatchSize != 0 ) 
			throw new IllegalArgumentException("numExamplesToFetch must be a multiple of miniBatchSize");
		if( miniBatchSize <= 0 ) throw new IllegalArgumentException("Invalid miniBatchSize (must be >0)");
		this.miniBatchSize = miniBatchSize;
		this.numExamplesToFetch = numExamplesToFetch;
		this.exampleLength = exampleLength;
		this.numCharacters = characters.length;
		this.characters = characters;
		this.rng = rng;
		this.alwaysRandomData = alwaysRandomData;
		this.artistId = new ArrayList<String>();
		this.file = new ArrayList<List<Integer>>();
		this.realFile = new ArrayList<List<Integer>>();
		this.actionNumber = new ArrayList<Double>();
		this.nextArtistId = null;
		Scanner sin = new Scanner(new File(textFilePath));
		while(sin.hasNext()){
			String[] line = sin.nextLine().split(",");
			List<Integer> sentence = new ArrayList<Integer>();
			List<Integer> realSentence = new ArrayList<Integer>();
			if(validArtistId != null && !validArtistId.equals(line[0]))
				continue;
//			System.out.println(line[0] + " " + validArtistId);
			this.artistId.add(line[0]);
			double action = 0;
			for(int i = 1; i < line.length; i++){
				sentence.add(characterToIndex(Double.parseDouble(line[i])));
				realSentence.add((int) Double.parseDouble(line[i]));
				action += Double.parseDouble(line[i]);
			}
			this.actionNumber.add(Math.sqrt(action));
			this.file.add(sentence);
			this.realFile.add(realSentence);
		}
		sin.close();
		System.out.println("Loaded and converted file: " + this.artistId.size());
	}
	
	private static int[] getDefultCharacters(int maxValue) {
		// TODO Auto-generated method stub
		if(maxValue < 1)
			throw new NumberFormatException("maxValue = "+maxValue+" < 1");
		List<Integer> res = new ArrayList<Integer>();
		int value = 1;
		while(value < maxValue){
			res.add(value);
			value = (int) (value*1.2+0.99);
		}
		int[] out = new int[res.size()];
		for(int i = 0; i < out.length; i++)
			out[i] = res.get(i);
		return out;
	}
	
	public int characterToIndex(double d){
		for(int i = 1; i < characters.length; i++){
			if(Math.abs(characters[i]-d) > Math.abs(characters[i-1]-d))
				return i-1;
		}
		return characters.length-1;
	}
	
	private int getRandomArtistId(){
		double sum = 0, d = rng.nextDouble();
		for(double action : actionNumber){
			sum += action;
		}
		for(int id = 0; id < actionNumber.size(); id++){
			if(d <= actionNumber.get(id)/sum)
				return id;
			d -= actionNumber.get(id)/sum;
		}
		throw new IllegalArgumentException("Distribution is invalid? sum="+sum);
	}
	
	public boolean hasNext() {
		if(!this.alwaysRandomData)
			return examplesSoFar + miniBatchSize <= this.file.size();
		return examplesSoFar + miniBatchSize <= numExamplesToFetch;
	}
	
	public List<String> getNextArtistId(){
		return nextArtistId;
	}
	
	public int[][] getRealOutput(){
		return realOutput;
	}
	
	public DataSet next() {
		return next(miniBatchSize);
	}

	public DataSet next(int num) {
		if(!this.alwaysRandomData && examplesSoFar+num>this.file.size())
			throw new NoSuchElementException();
		if( examplesSoFar+num > numExamplesToFetch ) 
			throw new NoSuchElementException();
		
		//Allocate space:
		INDArray input = Nd4j.zeros(num,numCharacters,exampleLength);
		INDArray labels = Nd4j.zeros(num,numCharacters,exampleLength);
				
		//Randomly select a subset of the file. No attempt is made to avoid overlapping subsets
		// of the file in the same minibatch
		nextArtistId = new ArrayList<String>();
		realOutput = new int[num][exampleLength];
		for( int i=0; i<num; i++ ){
			int artistIdx = getRandomArtistId();
			if(this.alwaysRandomData == false)
				artistIdx = examplesSoFar+i;
			if(file.get(artistIdx).size()<exampleLength+1)
				throw new NoSuchElementException("artistId = "+artistId.get(artistIdx)+" length < "+exampleLength+1);
			nextArtistId.add(artistId.get(artistIdx));
			int startIdx = (int) (rng.nextDouble()*(file.get(artistIdx).size()-exampleLength-1));
			if(this.alwaysRandomData == false)
				startIdx = 0;
			int currCharIdx = file.get(artistIdx).get(startIdx);
			for( int j=0; j<exampleLength; j++ ){
				int nextCharIdx = file.get(artistIdx).get(startIdx+j+1);
				input.putScalar(new int[]{i,currCharIdx,j}, 1.0);
				labels.putScalar(new int[]{i,nextCharIdx,j}, 1.0);
				realOutput[i][j] = realFile.get(artistIdx).get(startIdx+j+1);
				currCharIdx = nextCharIdx;
			}
		}
		
		examplesSoFar += num;
		return new DataSet(input,labels);
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

	public int getCharacterValue(int index){
		if(index >= characters.length)
			throw new RuntimeException("index="+index+" > characters.length="+characters.length);
		return characters[index];
	}
	
	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}

}