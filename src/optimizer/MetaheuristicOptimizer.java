package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;
import pso.Solution;

public abstract class MetaheuristicOptimizer extends Optimizer
{
	protected Solution globalBestSolution;
	protected double globalBestValue;

	protected Layer[] evaluateLayers;

	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);

		double[][][] weight = new double[layers.length][][];
		double[][] bias = new double[layers.length][];

		evaluateLayers = new Layer[layers.length];

		for (int i = 0; i < layers.length; i++)
		{
			weight[i] = layers[i].getWeight();
			bias[i] = layers[i].getBias();
			evaluateLayers[i] = new Layer(layers[i]);
		}

		globalBestSolution = new Solution();
		globalBestSolution.setWeight(weight);
		globalBestSolution.setBias(bias);
	}
}
