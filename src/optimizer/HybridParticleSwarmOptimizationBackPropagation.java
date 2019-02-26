package optimizer;

import activation.AbstractActivation;
import activation.Differentiable;
import layer.Layer;
import loss.AbstractLossFunction;
import pso.Particle;
import pso.Solution;

// PSO-BP
// https://www.sciencedirect.com/science/article/pii/S0096300306008277
public class HybridParticleSwarmOptimizationBackPropagation extends BatchParticleSwarmOptimization
{
	public static final int FIRST_CONDITION = 0;
	public static final int SECOND_CONDITION = 1;

	private int mode;
	private int bpSearchGenerations;
	private int bpCount;

	private int psoGenerations;
	private int psoCount;

	private double learningRate;
	private double originLearningRate;
	private double learningRateDecayRate;

	private boolean switchToBP;
	private int psoGlobalSolutionFixCount;
	private double previousPsoGlobalValue;

	private Particle worstParticle;
	private double worstValue;

	private double[][] guessArray;

	public HybridParticleSwarmOptimizationBackPropagation(pso.Parameter psoParameter, int batch, int condition,
			int bpSearchGenerations, int psoGenerations, double learningRate, double learningRateDecayRate)
	{
		super(psoParameter, batch);
		this.mode = condition;
		this.bpSearchGenerations = bpSearchGenerations;
		this.bpCount = 0;
		this.psoGenerations = psoGenerations;
		this.psoCount = 0;
		this.originLearningRate = learningRate;
		this.learningRateDecayRate = learningRateDecayRate;
		this.switchToBP = false;
		this.psoGlobalSolutionFixCount = 0;
		this.previousPsoGlobalValue = 0;
		this.worstParticle = null;
		this.worstValue = 0;
		this.guessArray = new double[batchSize][];
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);

		for (int i = 0; i < batchSize; i++)
		{
			guessArray[i] = new double[layers[layers.length - 1].getNeuronSize()];
		}
	}

	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		updateCount = updateCount + 1;
		if (updateCount < pso.linearEndCount)
		{
			w = pso.maxW - (pso.linearEndW / pso.linearEndCount) * updateCount;
		}
		else
		{
			w = (pso.maxW - pso.linearEndW) * Math.exp((pso.linearEndCount - updateCount) / pso.nonlinearlyWeight);
		}

		if (batchCount < batchSize)
		{
			double[] feature = layers[0].getInput();

			saveValueToArray(featureArray, feature, batchCount);
			saveValueToArray(labelArray, trueValue, batchCount);
			saveValueToArray(guessArray, guessValue, batchCount);

			batchCount = batchCount + 1;
		}

		if (batchCount >= batchSize)
		{
			batchUpdate();
		}
	}

	@Override
	public void batchUpdate()
	{
		if (switchToBP == false)
		{
			psoUpdate();

			switch (mode)
			{
				case FIRST_CONDITION:
					firstCondition();
					break;
				case SECOND_CONDITION:
					secondCondition();
					break;
				default:
					throw new UnsupportedOperationException("PSO-BP only two condition");
			}
		}
		else
		{
			bpUpdate();
		}
		// BS-IPSO-BP open, IPSO-BP close
		resetBatch();
	}

	private void psoUpdate()
	{
		psoCount = psoCount + 1;

		evaluate(featureArray, labelArray);
		determine();
		transit();
		evaluate(featureArray, labelArray);
		determine();

		double[][][] weight = globalBestSolution.getWeight();
		double[][] bias = globalBestSolution.getBias();
		for (int i = 0; i < layers.length; i++)
		{
			layers[i].updateWeight(weight[i]);
			layers[i].updateBias(bias[i]);
		}
	}

	private void firstCondition()
	{
		if (previousPsoGlobalValue == 0)
		{
			previousPsoGlobalValue = globalBestValue;
		}
		else if (previousPsoGlobalValue == globalBestValue)
		{
			psoGlobalSolutionFixCount = psoGlobalSolutionFixCount + 1;
		}
		else
		{
			psoGlobalSolutionFixCount = 0;
			previousPsoGlobalValue = globalBestValue;
		}

		if (psoGlobalSolutionFixCount >= 10 || psoCount >= psoGenerations)
		{
			switchToBP = true;
			psoCount = 0;
			psoGlobalSolutionFixCount = 0;
			previousPsoGlobalValue = 0;
		}
	}

	private void secondCondition()
	{
		switchToBP = true;
	}

	private void bpUpdate(double[] guessValue, double[] trueValue)
	{
		double[] previousError;
		double[] nowError;
		double error = 0;

		bpCount = bpCount + 1;
		worstValue = 0;

		setSolutionWeightToLayers(globalBestSolution);
		nowError = lossFunction.toDifferentiate(guessValue, trueValue);
		for (int i = evaluateLayers.length - 1; i >= 0; i--)
		{
			previousError = activationBackPropagation(nowError, evaluateLayers[i]);
			nowError = calculateError(previousError, evaluateLayers[i].getWeight());
			update(evaluateLayers[i], previousError, evaluateLayers[i].getInput());
		}

		for (int i = 0; i < featureArray.length; i++)
		{
			error = error + evaluate(featureArray[i], labelArray[i]);
		}

		for (int i = 0; i < particle.length; i++)
		{
			if (worstValue < particle[i].getLocalBestValue() || worstValue == 0)
			{
				worstParticle = particle[i];
				worstValue = particle[i].getLocalBestValue();
			}
		}

		if (error < globalBestValue)
		{
			double[][][] weight = new double[evaluateLayers.length][][];
			double[][] bias = new double[evaluateLayers.length][];

			for (int i = 0; i < evaluateLayers.length; i++)
			{
				weight[i] = evaluateLayers[i].getWeight();
				bias[i] = evaluateLayers[i].getBias();
			}

			globalBestValue = error;
			globalBestSolution = new Solution(weight, bias);

			for (int i = 0; i < layers.length; i++)
			{
				layers[i].updateWeight(weight[i]);
				layers[i].updateBias(bias[i]);
			}
		}
		else if (error < worstValue)
		{
			double[][][] weight = new double[evaluateLayers.length][][];
			double[][] bias = new double[evaluateLayers.length][];

			for (int i = 0; i < evaluateLayers.length; i++)
			{
				weight[i] = evaluateLayers[i].getWeight();
				bias[i] = evaluateLayers[i].getBias();
			}

			Solution solution = new Solution(weight, bias);
			worstParticle.setLocalBestSolution(solution);
			worstParticle.setLocalBestValue(error);
			worstParticle.setNowSolution(solution);
			worstParticle.setNowValue(error);
			setVelocity(worstParticle, weight, bias);
		}

		if (bpCount >= bpSearchGenerations)
		{
			switchToBP = false;
			bpCount = 0;
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
	public void newEpoch(int currentEpoch)
	{
		learningRate = originLearningRate * Math.exp(-1 * learningRateDecayRate * currentEpoch);
	}
}
