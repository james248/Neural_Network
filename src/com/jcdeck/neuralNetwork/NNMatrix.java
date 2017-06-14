package com.jcdeck.neuralNetwork;

import com.jcdeck.matrix.Matrix;

/**
 * 
 * This class extends the Matrix class to include a normalization
 * function and contains more accurately named "getter" methods.
 * Each row in the matrix should be a unique test case and the
 * columns should contain the inputs/outputs of that case.
 * 
 * @author James Decker
 *
 */
public class NNMatrix extends Matrix{
	
	//data = [numOfTestCases][numOfInputs/outputs]
	
	/**
	 * Constructs a new matrix with values the values in the
	 * two dimensional array passed.
	 * 
	 * @param d data to construct the matrix with
	 */
	public NNMatrix(double[][] d){
		super(d);
	}
	
	/**
	 * Constructs a new matrix with m rows and n columns. Each element
	 * is set to 0.
	 * 
	 * @param m The number of rows for the matrix
	 * @param n tHe number of columns for the matrix
	 */
	public NNMatrix(int m, int n){
		super(m, n);
	}
	
	/**
	 * 
	 * Divides each column of the matrix by the max value of that column
	 * 
	 */
	public void normilize(){
		
		//cycle through each column to normilize all of them
		for(int i = 0; i<n; i++){
			//find the max value of the column
			double max = data[0][i];
			for(int j = 0; j<m; j++)
				max = Math.max(max, data[j][i]);
			//divide each element by the max
			for(int j = 0; j<m; j++)
				data[j][i] /= max;
		}
		
	}
	
	
	/**
	 * Returns the number of test cases in a matrix. This is the same as
	 * the number of rows.
	 * 
	 * @return The number of test cases.
	 */
	public int getNumOfTestCases(){
		return super.getM();
	}
	/**
	 * Returns the number of nodes in that example set of input.output
	 * data. It is the same as the number of columns.
	 * 
	 * @return The number of nodes in the test case.
	 */
	public int getNumOfNodes(){
		return super.getN();
	}
	
	
}
