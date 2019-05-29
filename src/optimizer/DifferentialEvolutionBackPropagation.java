package optimizer;

import activation.AbstractActivation;
import activation.Differentiable;
import layer.Layer;
import loss.AbstractLossFunction;

public class DifferentialEvolutionBackPropagation extends MetaheuristicOptimizer
{
	private int deGenerations;
	private int deCount;

	private double learningRate;
	private double originLearningRate;
	private double learningRateDecayRate;

	// layer neurons weight
	private double[][][] weightDeltaArray;
	// layer bias
	private double[][] biasDeltaArray;

	public DifferentialEvolutionBackPropagation(int dataSize, int deGenerations, double learningRate,
			double learningRateDecayRate)
	{
		super(dataSize);
		this.deGenerations = deGenerations;
		this.deCount = 0;
		this.originLearningRate = learningRate;
		this.learningRateDecayRate = learningRateDecayRate;
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);
	}

	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		super.update(guessValue, trueValue);
	}

	@Override
	public void update()
	{
		if (deCount < deGenerations)
		{
			deUpdate();
		}
		else
		{
			bpUpdate();
		}
		reset();
	}

	public void deUpdate()
	{

	}

	private void bpUpdate()
	{
		double[] previousError;
		double[] nowError;
		double[] guessValue;

		setSolutionWeightToLayers(globalBestSolution);
		for (int i = 0; i < featureArray.length; i++)
		{
			guessValue = predict(featureArray[i]);
			nowError = lossFunction.toDifferentiate(guessValue, labelArray[i]);
			for (int j = evaluateLayers.length - 1; j >= 0; j--)
			{
				previousError = activationBackPropagation(nowError, evaluateLayers[j]);
				nowError = calculateError(previousError, evaluateLayers[j].getWeight());
				calculateDeltaValue(previousError, evaluateLayers[j].getInput(), j);
			}
		}

		for (int i = 0; i < evaluateLayers.length; i++)
		{
			update(layers[i], i);
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
		previousDataOutput = previousLayer.previousActivationOutput();
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

	private void calculateDeltaValue(double[] error, double[] previousDataOutput, int nowLayer)
	{
		for (int i = 0; i < error.length; i++)
		{
			for (int j = 0; j < previousDataOutput.length; j++)
			{
				weightDeltaArray[nowLayer][i][j] = weightDeltaArray[nowLayer][i][j]
						+ learningRate * error[i] * previousDataOutput[j];
			}
			biasDeltaArray[nowLayer][i] = biasDeltaArray[nowLayer][i] + learningRate * error[i];
		}
	}

	private void update(Layer layer, int nowLayer)
	{
		double[][] weight = layer.getWeight();
		double[] bias = layer.getBias();
		for (int i = 0; i < weightDeltaArray[nowLayer].length; i++)
		{
			for (int j = 0; j < weightDeltaArray[nowLayer][i].length; j++)
			{
				weightDeltaArray[nowLayer][i][j] = weightDeltaArray[nowLayer][i][j] / dataCount;
				weight[i][j] = weight[i][j] - weightDeltaArray[nowLayer][i][j];
			}
		}

		for (int i = 0; i < biasDeltaArray[nowLayer].length; i++)
		{
			biasDeltaArray[nowLayer][i] = biasDeltaArray[nowLayer][i] / dataCount;
			bias[i] = bias[i] - biasDeltaArray[nowLayer][i];
		}
		layer.updateWeight(weight);
		layer.updateBias(bias);
	}

	public void reset()
	{
		for (int i = 0; i < layers.length; i++)
		{
			for (int j = 0; j < weightDeltaArray[i].length; j++)
			{
				for (int k = 0; k < weightDeltaArray[i][j].length; k++)
				{
					weightDeltaArray[i][j][k] = 0;
				}
			}

			for (int j = 0; j < biasDeltaArray[i].length; j++)
			{
				biasDeltaArray[i][j] = 0;
			}
		}
	}

	@Override
	public void newEpoch(int currentEpoch)
	{
		learningRate = originLearningRate * Math.exp(-1 * learningRateDecayRate * currentEpoch);
	}
}
