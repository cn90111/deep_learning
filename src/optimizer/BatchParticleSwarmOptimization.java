package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;
import pso.Particle;
import pso.Solution;

public class BatchParticleSwarmOptimization extends AbstractOptimizer implements SupportBatchUpdate
{
	private double c1;
	private double c2;
	private double maxW;
	private double linearEndW;
	private double w;
	private double velocityLimit;
	private Solution globalBestSolution;
	private double globalBestValue;
	private Particle[] particle;
	private Layer[] layers;
	private Layer[] evaluateLayers;
	private double updateCount;
	private double linearEndCount;
	private double k;
	private AbstractLossFunction lossFunction;

	private int batchSize;
	private int batchCount;
	private double[][] featureArray;
	private double[][] labelArray;

	public BatchParticleSwarmOptimization(int particleSize, double globalSearchWeight, double localSearchWeight,
			double velocityRate, double velocityLimit, double solutionLimit, double linearEndCount,
			double nonlinearlyWeight)
	{
		this(particleSize, globalSearchWeight, localSearchWeight, velocityRate, velocityLimit, solutionLimit,
				linearEndCount, nonlinearlyWeight, 1);
	}

	public BatchParticleSwarmOptimization(int particleSize, double globalSearchWeight, double localSearchWeight,
			double velocityRate, double velocityLimit, double solutionLimit, double linearEndCount,
			double nonlinearlyWeight, int batch)
	{
		this.c1 = localSearchWeight;
		this.c2 = globalSearchWeight;
		this.maxW = velocityRate;
		this.velocityLimit = velocityLimit;
		this.linearEndCount = linearEndCount;
		this.k = nonlinearlyWeight;
		this.batchSize = batch;

		featureArray = new double[batchSize][];
		labelArray = new double[batchSize][];

		updateCount = 0;
		batchCount = 0;
		linearEndW = maxW / 2; // http://www.cse.cuhk.edu.hk/~lyu/paper_pdf/sdarticle.pdf

		particle = new Particle[particleSize];

		for (int i = 0; i < particle.length; i++)
		{
			particle[i] = new Particle(velocityLimit, solutionLimit);
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

		for (int i = 0; i < batchSize; i++)
		{
			featureArray[i] = new double[layers[0].getLinkSize()];
			labelArray[i] = new double[layers[layers.length - 1].getNeuronSize()];
		}

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

	private void setVelocity(Particle particle, double[][][] weight, double[][] bias)
	{
		double[][][] weightVelocity = new double[weight.length][][];
		double[][] biasVelocity = new double[bias.length][];
		for (int i = 0; i < weightVelocity.length; i++)
		{
			weightVelocity[i] = new double[weight[i].length][];
			for (int j = 0; j < weightVelocity[i].length; j++)
			{
				weightVelocity[i][j] = new double[weight[i][j].length];
				for (int k = 0; k < weightVelocity[i][j].length; k++)
				{
					weightVelocity[i][j][k] = Math.random() * (velocityLimit / 10) * 2 - (velocityLimit / 10);
				}
			}
		}
		for (int i = 0; i < biasVelocity.length; i++)
		{
			biasVelocity[i] = new double[bias[i].length];
			for (int j = 0; j < biasVelocity[i].length; j++)
			{
				biasVelocity[i][j] = Math.random() * (velocityLimit / 10) * 2 - (velocityLimit / 10);
			}
		}
		particle.setVelocity(new Solution(weightVelocity, biasVelocity));
	}

	private void setSolution(Particle particle, double[][][] weight, double[][] bias)
	{
		double[][][] solutionWeight = new double[weight.length][][];
		double[][] solutionBias = new double[bias.length][];
		for (int i = 0; i < solutionWeight.length; i++)
		{
			solutionWeight[i] = new double[weight[i].length][];
			for (int j = 0; j < solutionWeight[i].length; j++)
			{
				solutionWeight[i][j] = new double[weight[i][j].length];
				for (int k = 0; k < solutionWeight[i][j].length; k++)
				{
					solutionWeight[i][j][k] = (Math.random() * 2) - 1; // 1 ~ -1;
				}
			}
		}
		for (int i = 0; i < solutionBias.length; i++)
		{
			solutionBias[i] = new double[bias[i].length];
			for (int j = 0; j < solutionBias[i].length; j++)
			{
				solutionBias[i][j] = (Math.random() * 2) - 1; // 1 ~ -1;
			}
		}
		particle.setNowSolution(new Solution(solutionWeight, solutionBias));
		particle.setLocalBestSolution(new Solution(solutionWeight, solutionBias));
	}

	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		updateCount = updateCount + 1;
		if (updateCount < linearEndCount)
		{
			w = maxW - (linearEndW / linearEndCount) * updateCount;
		}
		else
		{
			w = (maxW - linearEndW) * Math.exp((linearEndCount - updateCount) / k);
		}
		double[] feature = layers[0].getInput();

		saveValueToArray(featureArray, feature, batchCount);
		saveValueToArray(labelArray, trueValue, batchCount);
		
		batchCount = batchCount + 1;

		if (batchCount >= batchSize)
		{
			batchUpdate();
		}
	}

	private void saveValueToArray(double[][] array, double[] value, int index)
	{
		for (int i = 0; i < value.length; i++)
		{
			array[index][i] = value[i];
		}
	}

	private void transit()
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
							+ c1 * localRandom * (localBestWeight[i][j][k] - nowWeight[i][j][k])
							+ c2 * globalRandom * (globalBestWeight[i][j][k] - nowWeight[i][j][k]);
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
				biasVelocity[i][j] = w * biasVelocity[i][j] + c1 * localRandom * (localBestBias[i][j] - nowBias[i][j])
						+ c2 * globalRandom * (globalBestBias[i][j] - nowBias[i][j]);
				nowBias[i][j] = nowBias[i][j] + biasVelocity[i][j];
			}
		}

		particle.setVelocity(new Solution(weightVelocity, biasVelocity));
		particle.setNowSolution(new Solution(nowWeight, nowBias));
	}

	private void evaluate(AbstractLossFunction lossFunction, double[][] feature, double[][] label)
	{
		// evaluate globalBestSolution loss with other feature input
		globalBestValue = evaluate(globalBestSolution, lossFunction, feature, label);

		for (int i = 0; i < particle.length; i++)
		{
			evaluate(particle[i], lossFunction, feature, label);
		}
	}

	private void evaluate(Particle particle, AbstractLossFunction loss, double[][] feature, double[][] label)
	{
		double error = evaluate(particle.getNowSolution(), loss, feature, label);

		if (error < particle.getLocalBestValue() || particle.getLocalBestValue() == 0)
		{
			particle.setLocalBestValue(error);
			particle.setLocalBestSolution(particle.getNowSolution());
		}
	}

	private double evaluate(Solution solution, AbstractLossFunction loss, double[][] feature, double[][] label)
	{
		double[][][] weight = solution.getWeight();
		double[][] bias = solution.getBias();
		double[] predictLabel;
		double error = 0;

		for (int i = 0; i < evaluateLayers.length; i++)
		{
			evaluateLayers[i].updateWeight(weight[i]);
			evaluateLayers[i].updateBias(bias[i]);
		}

		for (int i = 0; i < feature.length; i++)
		{
			predictLabel = predict(feature[i]);
			error = error + loss.getError(predictLabel, label[i]);
		}

		return error;
	}

	private void determine()
	{
		for (int i = 0; i < particle.length; i++)
		{
			if (particle[i].getLocalBestValue() < globalBestValue)
			{
				globalBestValue = particle[i].getLocalBestValue();
				globalBestSolution = particle[i].getLocalBestSolution();
			}
		}
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

	@Override
	public void batchUpdate()
	{
		evaluate(lossFunction, featureArray, labelArray);
		determine();
		transit();
		evaluate(lossFunction, featureArray, labelArray);
		determine();

		double[][][] weight = globalBestSolution.getWeight();
		double[][] bias = globalBestSolution.getBias();
		for (int i = 0; i < layers.length; i++)
		{
			layers[i].updateWeight(weight[i]);
			layers[i].updateBias(bias[i]);
		}

		resetBatch();
	}

	@Override
	public int getBatchSize()
	{
		return batchSize;
	}

	@Override
	public void resetBatch()
	{
		batchCount = 0;

		for (int i = 0; i < featureArray.length; i++)
		{
			for (int j = 0; j < featureArray[i].length; j++)
			{
				featureArray[i][j] = 0;
			}

			for (int j = 0; j < labelArray[i].length; j++)
			{
				labelArray[i][j] = 0;
			}
		}
	}
}
