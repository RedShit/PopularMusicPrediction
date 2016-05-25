package com.thinkingmaze.regression.tianchi;


import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/** A very simple DataSetIterator for use in the GravesLSTMCharModellingExample.
 * Given a text file and a few options, generate feature vectors and labels for training,
 * where we want to predict the next character in the sequence.<br>
 * This is done by randomly choosing a position in the text file to start the sequence and
 * (optionally) scanning backwards to a new line (to ensure we don't start half way through a word
 * for example).<br>
 * Feature vectors and labels are both one-hot vectors of same length
 * @author Alex Black
 */
public class ArtistIterator {
	private List<List<Integer>> file;
	private List<String> artistId;
	private int examplesSoFar;

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
	public ArtistIterator(String textFilePath,String validArtistId ) throws IOException {
		if( !new File(textFilePath).exists()) 
			throw new IOException("Could not access file (does not exist): " + textFilePath);
		this.artistId = new ArrayList<String>();
		this.file = new ArrayList<List<Integer>>();
		this.examplesSoFar = 0;
		Scanner sin = new Scanner(new File(textFilePath));
		while(sin.hasNext()){
			String[] line = sin.nextLine().split(",");
			List<Integer> sentence = new ArrayList<Integer>();
			if(validArtistId != "all" && !validArtistId.equals(line[0]))
				continue;
//			System.out.println(line[0] + " " + validArtistId);
			this.artistId.add(line[0]);
			for(int i = 1; i < line.length; i++){
				sentence.add(Integer.parseInt(line[i]));
			}
			this.file.add(sentence);
		}
		sin.close();
		if(this.artistId.size() == 0)
			throw new IOException("Could not access artist does not exist");
		System.out.println("Loaded and converted file: " + this.artistId.size());
	}
	
	public List<Integer> getActionList(String id){		
		for(int i = 0; i < artistId.size(); i++){
			if(artistId.get(i).equals(id))
				return file.get(i);
		}
		return null;
	}

	public void reset() {
		examplesSoFar = 0;
	}

	public int cursor() {
		return examplesSoFar;
	}
	
	public void remove() {
		throw new UnsupportedOperationException();
	}

}