package optimizer;

import activation.AbstractActivation;
import activation.Differentiable;
import layer.Layer;
import loss.AbstractLossFunction;
import metaheuristic.Particle;
import metaheuristic.Solution;

// PSO-BP
// https://www.sciencedirect.com/science/article/pii/S0096300306008277
public class HybridParticleSwarmOptimizationBackPropagation extends AdjustmentParticleSwarmOptimization
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

	// layer neurons weight
	private double[][][] weightDeltaArray;
	// layer bias
	private double[][] biasDeltaArray;

	public HybridParticleSwarmOptimizationBackPropagation(metaheuristic.PsoParameter psoParameter, int dataSize,
			int condition, int bpSearchGenerations, int psoGenerations, double learningRate,
			double learningRateDecayRate)
	{
		super(psoParameter, dataSize);
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
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);

		this.weightDeltaArray = new double[layers.length][][];
		this.biasDeltaArray = new double[layers.length][];

		for (int i = 0; i < layers.length; i++)
		{
			weightDeltaArray[i] = layers[i].getWeight();
			biasDeltaArray[i] = layers[i].getBias();
		}

		reset();
	}

	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		super.update(guessValue, trueValue);
	}

	public void update()
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
		reset();
	}

	protected void psoUpdate()
	{
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
		psoCount = psoCount + 1;
	}

	protected void evaluate(double[][] feature, double[][] label)
	{
		if (globalBestValue == 0)
		{
			setSolutionWeightToLayers(globalBestSolution);
			for (int i = 0; i < feature.length; i++)
			{
				globalBestValue = globalBestValue + evaluate(feature[i], label[i]);
			}
		}
		for (int i = 0; i < particle.length; i++)
		{
			evaluate(particle[i], feature, label);
		}
	}

	private void evaluate(Particle particle, double[][] feature, double[][] label)
	{
		double lossValue = 0;
		setSolutionWeightToLayers(particle.getNowSolution());
		for (int i = 0; i < feature.length; i++)
		{
			lossValue = lossValue + evaluate(feature[i], label[i]);
		}
		particle.setNowValue(lossValue);
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

	private void bpUpdate()
	{
		double[] previousError;
		double[] nowError;
		double[] guessValue;
		double error = 0;

		bpCount = bpCount + 1;
		worstValue = 0;

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
			update(evaluateLayers[i], i);
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
			globalBestSolution.setWeight(weight);
			globalBestSolution.setBias(bias);

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

	@Override
	public void newEpoch(int currentEpoch)
	{
		learningRate = originLearningRate * Math.exp(-1 * learningRateDecayRate * currentEpoch);
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
}
