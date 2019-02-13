package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;
import pso.Particle;
import pso.Solution;

//PSO w auto Adjustment
//https://www.sciencedirect.com/sc ience/article/pii/S0096300306008277
public abstract class AdjustmentParticleSwarmOptimization extends Optimizer
{
	protected pso.Parameter pso;
	protected Solution globalBestSolution;
	protected double globalBestValue;
	protected Particle[] particle;
	protected Layer[] evaluateLayers;

	protected double w;
	protected int updateCount;

	public AdjustmentParticleSwarmOptimization(pso.Parameter psoParameter)
	{
		this.pso = psoParameter;

		updateCount = 0;
		particle = new Particle[pso.size];

		for (int i = 0; i < particle.length; i++)
		{
			particle[i] = new Particle(pso.velocityLimit, pso.solutionLimit);
		}
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		this.layers = layers;
		this.lossFunction = lossFunction;
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

		for (int i = 0; i < particle.length; i++)
		{
			particleInit(particle[i], weight, bias);
		}
	}

	private void particleInit(Particle particle, double[][][] weight, double[][] bias)
	{
		setVelocity(particle, weight, bias);
		setSolution(particle, weight, bias);
	}

	protected void setVelocity(Particle particle, double[][][] weight, double[][] bias)
	{
		double[][][] weightVelocity;
		double[][] biasVelocity;
		weightVelocity = randomSetValueTo3DArray(weight, pso.initVelocityUpperLimit, pso.initVelocitylowerLimit);
		biasVelocity = randomSetValueTo2DArray(bias, pso.initVelocityUpperLimit, pso.initVelocitylowerLimit);
		particle.setVelocity(new Solution(weightVelocity, biasVelocity));
	}

	protected void setSolution(Particle particle, double[][][] weight, double[][] bias)
	{
		double[][][] solutionWeight = null;
		double[][] solutionBias = null;
		solutionWeight = randomSetValueTo3DArray(weight, pso.initVelocityUpperLimit, pso.initVelocitylowerLimit);
		solutionBias = randomSetValueTo2DArray(bias, pso.initVelocityUpperLimit, pso.initVelocitylowerLimit);
		particle.setNowSolution(new Solution(solutionWeight, solutionBias));
		particle.setLocalBestSolution(new Solution(solutionWeight, solutionBias));
	}

	private double[][][] randomSetValueTo3DArray(double[][][] arraySize, double upperLimit, double lowerLimit)
	{
		double[][][] array = new double[arraySize.length][][];
		for (int i = 0; i < array.length; i++)
		{
			array[i] = randomSetValueTo2DArray(arraySize[i], upperLimit, lowerLimit);
		}
		return array;
	}

	private double[][] randomSetValueTo2DArray(double[][] arraySize, double upperLimit, double lowerLimit)
	{
		double[][] array = new double[arraySize.length][];
		for (int i = 0; i < array.length; i++)
		{
			array[i] = new double[arraySize[i].length];
			for (int j = 0; j < array[i].length; j++)
			{
				array[i][j] = Math.random() * (upperLimit - lowerLimit) + lowerLimit; // upperLimit ~ lowerLimit
			}
		}
		return array;
	}

	protected void transit()
	{
		double[][][] globalBestWeight = globalBestSolution.getWeight();
		double[][] globalBestBias = globalBestSolution.getBias();

		for (int i = 0; i < particle.length; i++)
		{
			transit(particle[i], globalBestWeight, globalBestBias);
		}
	}

	private void transit(Particle particle, double[][][] globalBestWeight, double[][] globalBestBias)
	{
		double[][][] weightVelocity = particle.getVelocity().getWeight();
		double[][] biasVelocity = particle.getVelocity().getBias();
		double[][][] nowWeight = particle.getNowSolution().getWeight();
		double[][] nowBias = particle.getNowSolution().getBias();
		double[][][] localBestWeight = particle.getLocalBestSolution().getWeight();
		double[][] localBestBias = particle.getLocalBestSolution().getBias();

		double localRandom;
		double globalRandom;

		for (int i = 0; i < weightVelocity.length; i++)
		{
			for (int j = 0; j < weightVelocity[i].length; j++)
			{
				for (int k = 0; k < weightVelocity[i][j].length; k++)
				{
					localRandom = (Math.random() * 2) - 1; // 1 ~ -1
					globalRandom = (Math.random() * 2) - 1; // 1 ~ -1
					weightVelocity[i][j][k] = w * weightVelocity[i][j][k]
							+ pso.c1 * localRandom * (localBestWeight[i][j][k] - nowWeight[i][j][k])
							+ pso.c2 * globalRandom * (globalBestWeight[i][j][k] - nowWeight[i][j][k]);
					nowWeight[i][j][k] = nowWeight[i][j][k] + weightVelocity[i][j][k];
				}
			}
		}

		for (int i = 0; i < biasVelocity.length; i++)
		{
			for (int j = 0; j < biasVelocity[i].length; j++)
			{
				localRandom = (Math.random() * 2) - 1; // 1 ~ -1
				globalRandom = (Math.random() * 2) - 1; // 1 ~ -1
				biasVelocity[i][j] = w * biasVelocity[i][j]
						+ pso.c1 * localRandom * (localBestBias[i][j] - nowBias[i][j])
						+ pso.c2 * globalRandom * (globalBestBias[i][j] - nowBias[i][j]);
				nowBias[i][j] = nowBias[i][j] + biasVelocity[i][j];
			}
		}

		particle.setVelocity(new Solution(weightVelocity, biasVelocity));
		particle.setNowSolution(new Solution(nowWeight, nowBias));
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

	private double[] predict(double[] feature)
	{
		evaluateLayers[0].dataIn(feature);
		for (int i = 1; i < evaluateLayers.length; i++)
		{
			evaluateLayers[i].dataIn(evaluateLayers[i - 1].dataOut());
		}
		return evaluateLayers[evaluateLayers.length - 1].dataOut();
	}

	protected void determine()
	{
		Particle temp;
		for (int i = 0; i < particle.length; i++)
		{
			temp = particle[i];
			if (temp.getNowValue() < temp.getLocalBestValue() || temp.getLocalBestValue() == 0)
			{
				temp.setLocalBestValue(temp.getNowValue());
				temp.setLocalBestSolution(temp.getNowSolution());
			}

			if (temp.getLocalBestValue() < globalBestValue)
			{
				globalBestValue = temp.getLocalBestValue();
				globalBestSolution = temp.getLocalBestSolution();
			}
		}
	}
}