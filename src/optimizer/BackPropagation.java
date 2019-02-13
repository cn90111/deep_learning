package optimizer;

import activation.AbstractActivation;
import activation.Differentiable;
import layer.Layer;
import loss.AbstractLossFunction;

// batch bp 
// https://stats.stackexchange.com/questions/174708/how-are-weights-updated-in-the-batch-learning-method-in-neural-networks
public class BackPropagation extends Optimizer implements SupportBatchUpdate
{
	private double learningRate;
	private int batchSize;
	private int batchCount;
	private double[][] sumErrorValue;
	public double updateCount;

	public BackPropagation(double learningRate)
	{
		this(learningRate, 1);
	}

	public BackPropagation(double learningRate, int batchSize)
	{
		this.learningRate = learningRate;
		this.batchSize = batchSize;
	}

	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		this.layers = layers;
		this.sumErrorValue = new double[layers.length][];
		this.lossFunction = lossFunction;
		for (int i = 0; i < layers.length; i++)
		{
			sumErrorValue[i] = new double[layers[i].getNeurons().length];
		}
		resetBatch();
	}

	@Override
	public void resetBatch()
	{
		batchCount = 0;
		for (int i = 0; i < layers.length; i++)
		{
			for (int j = 0; j < sumErrorValue[i].length; j++)
			{
				sumErrorValue[i][j] = 0;
			}
		}

	}

	@Override
	public void update(double guessValue[], double trueValue[])
	{
		double[] previousError;
		double[] nowError;

		batchCount = batchCount + 1;

		nowError = lossFunction.toDifferentiate(guessValue, trueValue);
		for (int i = layers.length - 1; i >= 0; i--)
		{
			previousError = activationBackPropagation(nowError, layers[i]);
			nowError = calculateError(previousError, layers[i].getWeight());

			for (int j = 0; j < previousError.length; j++)
			{
				sumErrorValue[i][j] = sumErrorValue[i][j] + previousError[j];
			}
		}

		if (batchCount >= batchSize)
		{
			batchUpdate();
		}
	}

	private double[] activationBackPropagation(double[] nowError, Layer previousLayer)
	{
		double[] previousDataOutput;
		double[] previousError = null;
		AbstractActivation previousActivation;

		previousActivation = previousLayer.getActivation();
		if (!(previousActivation instanceof Differentiable))
		{
			System.out.println("Activation can't Differential");
			System.exit(1);
		}
		previousDataOutput = previousLayer.dataOut();
		previousError = ((Differentiable) previousActivation).toDifferentiate(previousDataOutput);
		for (int i = 0; i < previousError.length; i++)
		{
			previousError[i] = previousError[i] * nowError[i];
		}
		return previousError;
	}

	private double[] calculateError(double[] error, double[][] weight)
	{
		double[] errorValue = new double[weight[0].length];
		for (int i = 0; i < errorValue.length; i++)
		{
			errorValue[i] = 0;
			for (int j = 0; j < error.length; j++)
			{
				errorValue[i] = errorValue[i] + error[j] * weight[j][i];
			}
		}
		return errorValue;
	}

	@Override
	public void batchUpdate()
	{
		for (int i = layers.length - 1; i >= 0; i--)
		{
			update(layers[i], sumErrorValue[i], layers[i].getInput());
		}
		resetBatch();
	}

	private void update(Layer layer, double[] error, double[] previousDataOutput)
	{
		double[][] weight = layer.getWeight();
		double[] bias = layer.getBias();
		for (int i = 0; i < error.length; i++)
		{
			for (int j = 0; j < previousDataOutput.length; j++)
			{
				weight[i][j] = weight[i][j] - learningRate * error[i] * previousDataOutput[j];
			}
			bias[i] = bias[i] - learningRate * error[i];
		}
		layer.updateWeight(weight);
		layer.updateBias(bias);
	}

	@Override
	public int getBatchSize()
	{
		return batchSize;
	}

	@Override
	public void newEpoch(int currentEpoch)
	{

	}
}
