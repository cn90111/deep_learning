package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;
import pso.Particle;
import pso.Solution;

//PSO w auto Adjustment
//https://www.sciencedirect.com/science/article/pii/S0096300306008277
public abstract class AdjustmentParticleSwarmOptimization extends MetaheuristicOptimizer
{
	protected pso.Parameter pso;
	protected Particle[] particle;

	protected double w;
	protected int updateCount;

	public AdjustmentParticleSwarmOptimization(pso.Parameter psoParameter, int dataSize)
	{
		super(dataSize);
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
		super.setConfiguration(layers, lossFunction);

		for (int i = 0; i < particle.length; i++)
		{
			particleInit(particle[i], globalBestSolution.getWeight(), globalBestSolution.getBias());
		}
	}
	
	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		updateInertia();

		super.update(guessValue, trueValue);
	}
	
	protected void updateInertia()
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
	}

	private void particleInit(Particle particle, double[][][] weightSize, double[][] biasSize)
	{
		setVelocity(particle, weightSize, biasSize);
		setSolution(particle, weightSize, biasSize);
	}

	protected void setVelocity(Particle particle, double[][][] weightSize, double[][] biasSize)
	{
		double[][][] weightVelocity;
		double[][] biasVelocity;
		weightVelocity = randomSetValueTo3DArray(weightSize, pso.initVelocityUpperLimit, pso.initVelocitylowerLimit);
		biasVelocity = randomSetValueTo2DArray(biasSize, pso.initVelocityUpperLimit, pso.initVelocitylowerLimit);
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
					localRandom = Math.random(); // 1 ~ 0
					globalRandom = Math.random(); // 1 ~ 0
					weightVelocity[i][j][k] = w * weightVelocity[i][j][k]
							+ pso.c1 * localRandom * (localBestWeight[i][j][k] - nowWeight[i][j][k])
							+ pso.c2 * globalRandom * (globalBestWeight[i][j][k] - nowWeight[i][j][k]);
				}
			}
		}

		for (int i = 0; i < biasVelocity.length; i++)
		{
			for (int j = 0; j < biasVelocity[i].length; j++)
			{
				localRandom = Math.random(); // 1 ~ 0
				globalRandom = Math.random(); // 1 ~ 0
				biasVelocity[i][j] = w * biasVelocity[i][j]
						+ pso.c1 * localRandom * (localBestBias[i][j] - nowBias[i][j])
						+ pso.c2 * globalRandom * (globalBestBias[i][j] - nowBias[i][j]);
			}
		}
		particle.setVelocity(new Solution(weightVelocity, biasVelocity));
		particle.updateSolution(particle.getVelocity());
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
