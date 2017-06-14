package com.jcdeck.neuralNetwork;

import java.util.ArrayList;

/**
 * Holds data about training the {@link Network}. Returned after the network has
 * been trained. Hold the time that it took to train the network, the number of
 * Iterations that occurred during training, and the final error of the network.
 * 
 * @author James C Decker
 *
 */
public class NNTrainingData {
	
	/**
	 * Constructs a new object to hold data from training the neural network.
	 */
	NNTrainingData(){
		J = new ArrayList<Double>();
	}
	
	//VARIABLES
	
	/**
	 * Time it took the network to be trained in milliseconds
	 */
	private double duration;
	
	/**
	 * The number of iterations it took the network to reduce the error to
	 * an acceptable amount
	 */
	private int iterations;
	
	/**
	 * The final error of the network
	 */
	private double finalError;
	
	/**
	 * the error at each iteration of the network
	 */
	private ArrayList<Double> J;
	
	//GETTERS AND SETTERS
	
	/**
	 * Sets the duration in milliseconds it took to train the network.
	 * 
	 * @param duration time it took to train the network in milliseconds
	 */
	void setDuration(double duration){
		this.duration = duration;
	}
	
	/**
	 * Returns the duration in milliseconds it took to train the network.
	 * 
	 * @return time it took to train the network in milliseconds
	 */
	public double getDuration(){
		return this.duration;
	}
	
	
	
	/**
	 * Sets the number of iterations that it took to train the network so
	 * that the error was an acceptable value. The number of iterations is the
	 * number of times forward and backward propagation occurred.
	 * 
	 * @param iterations the number of iteration taken when training the netwoRk
	 */
	void setIterations(int iterations){
		this.iterations = iterations;
	}
	
	/**
	 * Returns the number of iterations that it took to train the network so
	 * that the error was an acceptable value. The number of iterations is the
	 * number of times forward and backward propagation occurred.
	 * 
	 * @return the number of iteration taken when training the netwoRk
	 */
	public int getIterations(){
		return this.iterations;
	}
	
	
	
	/**
	 * Sets the final error of the network produced when the training matrix
	 * was propagated through it. Should be less that the maximum error acceptable
	 * unless the network hit the maximum number of iterations.
	 * 
	 * @param finalError error after training the network
	 */
	void finalError(double finalError){
		this.finalError = finalError;
	}
	
	/**
	 * Returns the final error of the network produced when the training matrix
	 * was propagated through it. Should be less that the maximum error acceptable
	 * unless the network hit the maximum number of iterations.
	 * 
	 * @return error after training the network
	 */
	public double getFinalError(){
		return this.finalError;
	}
	
	
	
	
	/**
	 * Adds {@code cost} to the list of cost values for each iteration.
	 * 
	 * @param cost cost of one iteration of forward propagation
	 */
	void addCostToGraph(double cost){
		this.J.add(cost);
	}
	
	/**
	 * Returns the error that occurred for each iteration of the
	 * training function. the length of the array is equal to the 
	 * number of iterations  made during training. [0] is the initial
	 * error and [length-1] is the final error after training.
	 * 
	 * @return array containing the errors for each iteration of training
	 */
	public double[] getCostGraph(){
		if(J == null)
			return new double[0];
		double[] d = new double[J.size()];
		for(int i = 0; i<d.length; i++)
			d[i] = J.get(i);
		return d;
	}
	
}
