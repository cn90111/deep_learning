package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;
import metaheuristic.Solution;

public abstract class MetaheuristicOptimizer extends Optimizer
{
	protected Solution globalBestSolution;
	protected double globalBestValue;

	protected Layer[] evaluateLayers;

	protected double[][] featureArray;
	protected double[][] labelArray;

	protected int dataSize;
	protected int dataCount;

	public MetaheuristicOptimizer(int dataSize)
	{
		this.dataSize = dataSize;
		this.dataCount = 0;
		featureArray = new double[dataSize][];
		labelArray = new double[dataSize][];
	}

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

		for (int i = 0; i < dataSize; i++)
		{
			featureArray[i] = new double[layers[0].getLinkSize()];
			labelArray[i] = new double[layers[layers.length - 1].getNeuronSize()];
		}
	}
	
	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		saveData(trueValue);

		if (dataCount >= dataSize)
		{
			update();
		}
	}
	
	public abstract void update();

	protected void saveData(double[] trueValue)
	{
		if (dataCount < dataSize)
		{
			double[] feature = layers[0].getInput();

			saveValueToArray(featureArray, feature, dataCount);
			saveValueToArray(labelArray, trueValue, dataCount);

			dataCount = dataCount + 1;
		}
	}

	protected void saveValueToArray(double[][] array, double[] value, int index)
	{
		for (int i = 0; i < value.length; i++)
		{
			array[index][i] = value[i];
		}
	}
	
	protected void setSolutionWeightToLayers(Solution solution)
	{
		double[][][] weight = solution.getWeight();
		double[][] bias = solution.getBias();

		for (int i = 0; i < evaluateLayers.length; i++)
		{
			evaluateLayers[i].updateWeight(weight[i]);
			evaluateLayers[i].updateBias(bias[i]);
		}
	}
	
	protected double evaluate(double[] feature, double[] label)
	{
		double[] predictLabel;
		double error;

		predictLabel = predict(feature);
		error = lossFunction.getError(predictLabel, label);

		return error;
	}

	protected double[] predict(double[] feature)
	{
		evaluateLayers[0].dataIn(feature);
		for (int i = 1; i < evaluateLayers.length; i++)
		{
			evaluateLayers[i].dataIn(evaluateLayers[i - 1].dataOutput());
		}
		return evaluateLayers[evaluateLayers.length - 1].dataOutput();
	}
}
