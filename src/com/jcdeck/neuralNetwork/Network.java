package com.jcdeck.neuralNetwork;

//import matrix.Matrix;

import com.jcdeck.matrix.Matrix;

/**
 * This neural network is designed to take a set of input and output values
 * and train the network so that it can predict the outputs for any given
 * set of input values.
 * 
 * @author James C Decker
 *
 */
public class Network {
	
	/**
	 * Constructs (but does not train) a new neural network based on the input and
	 * output matrixes passed. Both Matrixes should be normalized before
	 * being passed to the constructor.
	 * 
	 * @param input input matrix containing test cases
	 * @param hiddenLayerSize The number of nodes in the hidden layer. This
	 * should be about 2/3 the sum of the number of input and output nodes.
	 * @param output The output matrix containing test cases corresponding to
	 * the input matrix
	 */
	public Network(NNMatrix input, int hiddenLayerSize, NNMatrix output){
		
		this.trainningInput = input;
		this.trainningOutput = output;
		
		if(input != null && output != null){
			this.inputLayerSize = input.getNumOfNodes();
			this.hiddenLayerSize = hiddenLayerSize;
			this.outputLayerSize = output.getNumOfNodes();
		}else{
			this.inputLayerSize = 0;
			this.hiddenLayerSize = 0;
			this.outputLayerSize = 0;
		}
		
		//init weights
		this.weights1 = Matrix.randomGaussian(this.inputLayerSize, this.hiddenLayerSize);
		this.weights2 = Matrix.randomGaussian(this.hiddenLayerSize, this.outputLayerSize);
		
		weights1.print();
		weights2.print();
		
	}
	
	/**
	 * Constructs (but does not train) a new neural network based on the input and
	 * output matrixes passed. Both Matrixes should be normalized before
	 * being passed to the constructor.
	 * 
	 * @param input input matrix containing test cases
	 * @param hiddenLayerSize number of nodes in the hidden layer. This
	 * should be about 2/3 the sum of the number of input and output nodes.
	 * @param output output matrix containing test cases corresponding to
	 * the input matrix
	 */
	public Network(double[][] input, int hiddenLayerSize, double[][] output){
		
		this.trainningInput = new NNMatrix(input);
		this.trainningOutput = new NNMatrix(output);
		
		this.inputLayerSize = this.trainningInput.getNumOfNodes();
		this.hiddenLayerSize = hiddenLayerSize;
		this.outputLayerSize = this.trainningOutput.getNumOfNodes();
		
		//init weights
		this.weights1 = Matrix.randomGaussian(this.inputLayerSize, this.hiddenLayerSize);
		this.weights2 = Matrix.randomGaussian(this.hiddenLayerSize, this.outputLayerSize);
		
	}
	
	//TRAINING DATA
	
	/**
	 * the input data set used for training the network
	 */
	private final NNMatrix trainningInput;
	/**
	 * the output data set used for training the network
	 */
	private final NNMatrix trainningOutput;
	
	//SIZES OF THE LAYERS
	
	/**
	 * number of nodes in each of the input layer of the network
	 */
	private final int inputLayerSize;
	
	/**
	 * number of nodes in each of the first and only hidden layer of the network
	 */
	private final int hiddenLayerSize;
	
	/**
	 * number of nodes in each of the output layer of the network
	 */
	private final int outputLayerSize;
	
	
	//WEIGHTS
	
	/**
	 * the weights applied to a matrix at the first (input) layer
	 */
	private Matrix weights1;
	
	/**
	 * the weights applied to a matrix at the second (hidden) layer
	 */
	private Matrix weights2;
	
	
	
	//INTERMEDIATE MATRIES
	
	/**
	 * the result of applying {@code weights1} to the input matrix
	 */
	private Matrix z2;
	
	/**
	 * the matrix after applying the activation sigmoid function to {@code z2}
	 */
	private Matrix a2;
	
	/**
	 * the result of applying {@code weights2} to the activated matrix {@code a2}
	 */
	private Matrix z3;
	
	/**
	 * The output of the network
	 */
	private NNMatrix yHat;
	
	
	//COST
	
	/**
	 * This contains the cost functions for each layer of the network. It
	 * will be two elements long. Used to adjust the weights
	 */
	private Matrix[] costPrime;
	
	
	
	
	
	
	//TRAINING FUNCTION
	
	/**
	 * Performs forward and backward propagation on the network with the
	 * stored matrixes until the error is less than 'epsilon'. The learning
	 * rate will be multiplied by deltaAlpha after every step.
	 * 
	 * @param alpha initial learning rate. Normally between 1 and 10
	 * @param deltaAlpha rate at which {@code alpha} will be changed. Normally
	 * between 0.9999 and 0.99999999
	 * @param epsilon lowest acceptable value for the error
	 * @param  maxIterations maximum number of iterations that will be made when training the neural
	 * network. Prevents infinite loop.
	 */
	public NNTrainingData train(double alpha, double deltaAlpha, double epsilon, int maxIterations){
		
		System.out.println("Beginning training");
		
		//create a data object to return with information about the training
		NNTrainingData data = new NNTrainingData();
		
		//record the start time to calculate the duration of the training
		long startTime = System.nanoTime();
		
		int i;
		for(i = 0; i<maxIterations; i++){
			
			//forward propagate the input
			//this.forward(this.trainningInput);
			
			
			//get the error of the prediction
			final double J = this.costFunction(this.trainningInput, this.trainningOutput);
			data.addCostToGraph(J);
			
			//System.out.println("Training Neural Network... Iteration: "+i+", Cost: "+J);
			
			//calculate the derivative of the cost
			this.costFunctionPrime(this.trainningInput, this.trainningOutput);
			
			//adjust the weights
			this.adjustWeights(alpha);
			
			//if the weight (J) is an acceptable value (less than epsilon) then training is complete
			if(J < epsilon){
				break;
			}
			
			//adjust the learning rate
			alpha *= deltaAlpha;
			
		}
		
		
		long endTime = System.nanoTime();
		
		//record the duration of the operation in milliseconds
		data.setDuration((endTime - startTime) / 1000000);
		
		//debug
		System.out.println("Training Complete: ("+data.getDuration()+" ms)");
		System.out.println("  --> "+data.getIterations()+" iterations to get an error of "+data.getFinalError());
		System.out.println();
		
		//return data about the training
		return data;
		
	}
	
	
	//PREDITION / FORWARDING FUNCTIONS
	
	/**
	 * Calculates the output matrix of the given input data base on the weights.
	 * 
	 * @param x input Matrix
	 * @return a matrix containing the predicted output values given the input matrix
	 */
	public NNMatrix predict(Matrix input){
		
		this.z2 = input.times(weights1);
		this.a2 = this.getSigmoid(z2);
		this.z3 = a2.times(weights2);
		this.yHat = this.getSigmoid(z3);
		
		return this.yHat;
		
	}
	
	/**
	 * Propagates the input Matrix {@code x} forward through the neural network.
	 * It is the same as the prediction function
	 * 
	 * @param x input matrix
	 * @return a matrix containing the predicted output values given the input matrix
	 */
	private NNMatrix forward(Matrix x){
		return predict(x);
	}
	
	
	//SIGMOID FUNCITONS FOR FORWARD AND BACKWARD PROPAGATION
	
	//FORWARD PROPAGATION
	
	/**
	 * Applies an element wise activation function to the matrix b. Each element
	 * is passed through the sigmoid function 1 / (1 + e^(-x)).
	 * 
	 * @param b A matrix to activate
	 * @return The activation of each element of b
	 */
	private NNMatrix getSigmoid(Matrix b){
		//create a new matrix, c, that will have the result of b when the sigmoid function is applied
		NNMatrix c = new NNMatrix(b.getM(), b.getN());
		//pass each element of b into the sigmoid function and store it in c
		for(int i = 0; i<b.getM(); i++)
			for(int j = 0; j<b.getN(); j++)
				c.set( getSigmoid(b.get(i, j)), i, j);
		//return the resulting matrix
		return c;
	}
	
	/**
	 * Returns the result of the activation of x. It uses the sigmoid function
	 * 1 / (1 + e^(-x)).
	 * 
	 * @param x The input variable
	 * @return The result of the activation function
	 */
	private double getSigmoid(double x){
		return 1 / (1 + Math.exp(-x));
	}
	
	
	//BACKWARD PROPAGATION
	
	/**
	 * Applies the derivative of the activation function element wise to the
	 * input matrix b. This is used to indicate the direction to adjust the
	 * weights. The function applied to each element is e^(-x) / (1 + e^(-x))^(2).
	 * 
	 * @param b The input matrix to apply the derivative of the sigmoid function to
	 * @return The slope of each element of the activation function
	 */
	private NNMatrix getSigmoidPrime(Matrix b){
		//create a new matrix, c, that will have the result of b when the derivate of the sigmoid function is applied
		NNMatrix c = new NNMatrix(b.getM(), b.getN());
		//pass each element of b into the derivative of the sigmoid function and store it in c
		for(int i = 0; i<b.getM(); i++)
			for(int j = 0; j<b.getN(); j++)
				c.set(getSigmoidPrime(b.get(i, j)), i, j);
		//return c
		return c;
	}
	
	/**
	 * Returns the derivative of the sigmoid function to indicate the direction
	 * in which to adjust the weights. It applies the function e^(-x) / (1 + e^(-x))^(2).
	 * 
	 * @param x the input variable
	 * @return The derivative of the activateion function at x
	 */
	private double getSigmoidPrime(double x){
		return Math.exp(-x) / Math.pow((1 + Math.exp(-x)), 2);
	}
	
	
	
	//COST FUNCTIONS
	
	/**
	 * Computes cost for x and y with weights already stored in class. The cost
	 * is the sum of the squares of the cost for each test case.
	 * 
	 * @return the cost for all test inputs
	 */
	private double costFunction(Matrix X, Matrix y){

//		System.out.println("costFunction() Broken:");
		forward(X);
		
        //#J = 0.5*sum((y-self.yHat)**2)
		

//		System.out.println("  Matrix yHat:");
//		yHat.print();
//		
//
//		System.out.println("  Matrix y:");
//		y.print();
		
		//find the difference between the actual output and the output the Network predicted
		Matrix c = y.minus(this.yHat);

//		System.out.println("  Matrix c:");
//		c.print();
//		
		//square each error for each testing case
		Matrix d = c.raiseTo(2);
		

//		System.out.println("  Matrix d:");
//		d.print();
		
		//return the half the sum
		final double cost =  d.sum() / 2;
		return cost;
	}
	
	/**
	 * Computes Derivative of {@code weight1} and {@code weight2} for a given X and Y. The length
	 * of the array is two - equal to the number of weights. Each matrix in the array
	 * is the same dimensions as the matrix it is the derivative of.
	 * 
	 * @param x the training inputs
	 * @param y the outputs after forwarding {@code x} through the matrix
	 * @return An array of matrices representing cost
	 */
	private Matrix[] costFunctionPrime(Matrix X, Matrix y){
		
		this.forward(X);
		
		
		/* ##
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		dJdW2 = np.dot(self.a2.T, delta3)
		
		delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
		dJdW1 = np.dot(X.T, delta2)  
		
		return dJdW1, dJdW2
		*/
		
		Matrix c = y.minus(this.yHat);
		Matrix d = c.multiplyBy(-1);
		Matrix delta3 = d.multiplyBy(this.getSigmoidPrime(this.z3));
		Matrix dJdW2 = this.a2.transpose().times(delta3);
		
		Matrix delta2 = delta3.times(this.weights2.transpose()).multiplyBy(this.getSigmoidPrime(this.z2));
		Matrix dJdW1 = X.transpose().times(delta2);
		
		this.costPrime = new Matrix[] {dJdW1, dJdW2};
		
		return this.costPrime;
	}
	
	
	
	
	
	//ADJUSTING WEIGHTS
	
	/**
	 * This will adjust all the values in both weights1 and
	 * weights2 to minimize the cost function. It bases the
	 * adjustments on this.costPrime.
	 * 
	 */
	private void adjustWeights(double alpha){
		
		//adjust each weight in the positive direction if the derivative in negative
		// --> The slope of the cost is downward so a higher weight will decrease the cost
		
		//increase/decrease in relation to the activation value
		// --> higher activations affected the output more and should be adjusted more
		
		//Do this for each matrix of weights
		// --> the weights for each layer
		
		weights1 = weights1.minus(costPrime[0].multiplyBy(alpha));
		weights2 = weights2.minus(costPrime[1].multiplyBy(alpha));
		
	}
	
	/**
	 * Returns the two weight matrixes used in the neural network. When
	 * the network is constructed both are randomly initialized with
	 * gaussian probability.
	 * 
	 * @return the weights used by the neural network
	 */
	public Matrix[] getWeights(){
		return new Matrix[] {this.weights1, this.weights2};
	}
	
	
	
	
	//TESTING
//	public static void main(String[] args){
//		double[][] in = {
//				{3, 5},
//				{5, 1},
//				{10, 2}
//		};
//		NNMatrix input = new NNMatrix(in);
//		input.normilize();
//		double[][] out = {
//				{0.75},
//				{0.82},
//				{0.93}
//		};
//		NNMatrix output = new NNMatrix(out);
//		//Matrix output = outpu.multiplyBy(0.01);
//		
//		Network network = new Network(input, 3, output);
//		
//		System.out.println("\nInitial Predictions");
//		Matrix firstTest = network.predict(input);
//		firstTest.print();
//		System.out.println();
//		
//		NNTrainingData d = network.train(10, 0.999, 0.000001);
//		
//		System.out.println(d);
//		
//
//		System.out.println("\nFinal Predictions");
//		Matrix secondTest = network.predict(input);
//		secondTest.print();
//		System.out.println();
//		
//		NNMatrix test = new NNMatrix(new double[][] {{8/24.0, 3/24.0}});
//		
//		network.predict(test).print();
//		
//	}
	
	
}
