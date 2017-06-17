package com.jcdeck.neuralNetwork;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;

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
	 * Returns the number of iterations that it took to train the network so
	 * that the error was an acceptable value. The number of iterations is the
	 * number of times forward and backward propagation occurred.
	 * 
	 * @return the number of iteration taken when training the netwoRk
	 */
	public int getIterations(){
		return this.J.size();
	}
	
	
	
	/**
	 * Returns the final error of the network produced when the training matrix
	 * was propagated through it. Should be less that the maximum error acceptable
	 * unless the network hit the maximum number of iterations.
	 * 
	 * @return error after training the network
	 */
	public double getFinalError(){
		return this.J.get(J.size()-1);
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
	
	
	
	//JFRAME
	private JFrame costGraph;
	
	public void showCostGraph(int size){
		
		costGraph = new JFrame("Neural Network Cost Graph");
		JPanel panel = new JPanel();
		
		//max error
		double[] costs = this.getCostGraph();
		double maxError = 0;
		for(int i = 0; i<costs.length; i++)
			maxError = Math.max(maxError, costs[i]);
		
		BufferedImage bimage = new BufferedImage(size, size, BufferedImage.TYPE_INT_ARGB);
		Graphics2D g = bimage.createGraphics();
		g.setColor(Color.WHITE);
		g.fillRect(0, 0, size, size);
		size -= 10;
		g.setColor(Color.BLUE);
		g.setStroke(new BasicStroke(3));
		for(int i = 0; i<size-1; i++){
			final int index = (int) (((double)i/size)*costs.length);
			final int height1 = (int) ((costs[index]/maxError) * size);
			final int height2 = (int) ((costs[index+1]/maxError) * size);
			g.drawLine(i+5, size-height1+15, i+6, size-height2+15);
		}
		g.dispose();
		
		panel.add(new JLabel(new ImageIcon(bimage)));
		costGraph.add(panel);
		
		costGraph.validate();
		costGraph.repaint();
		costGraph.pack();
		costGraph.setVisible(true);
		
	}
	
	
	
	
	
	
	
	
	
	@Override
	public String toString(){
		return this.getIterations()+" iterations to get an error of "+this.getFinalError();
	}
	
}
