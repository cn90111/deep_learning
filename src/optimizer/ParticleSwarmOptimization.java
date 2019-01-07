package optimizer;

import java.util.ArrayList;

import layer.Layer;
import loss.AbstractLoss;
import pso.Particle;
import pso.Solution;

public class ParticleSwarmOptimization extends AbstractOptimizer
{
	private double c1;
	private double c2;
	private double maxW;
	private double w;
	private double velocityLimit;
	private Solution globalBestSolution;
	private double globalBestValue;
	private Particle[] particle;
	private Layer[] layers;
	private Layer[] evaluateLayers;
	private ArrayList<double[]> inputFeature;
	private ArrayList<double[]> outputLabel;
	private double updateCount;

	public ParticleSwarmOptimization(int particleSize, double globalSearchWeight, double localSearchWeight,
			double velocityRate, double velocityLimit, double solutionLimit)
	{
		this.c1 = localSearchWeight;
		this.c2 = globalSearchWeight;
		this.maxW = velocityRate;
		this.velocityLimit = velocityLimit;
		this.globalBestValue = 0;

		inputFeature = new ArrayList<double[]>();
		outputLabel = new ArrayList<double[]>();

		updateCount = 0;

		particle = new Particle[particleSize + 1]; // +1 save globalSolution

		for (int i = 0; i < particle.length; i++)
		{
			particle[i] = new Particle(velocityLimit, solutionLimit);
		}
	}

	@Override
	public void setLayers(Layer[] layers)
	{
		this.layers = layers;
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
	public void update(AbstractLoss loss, double[] guessValue, double[] trueValue)
	{
		updateCount = updateCount + 1;
		w = maxW / updateCount;
		double[] feature = layers[0].getInput();
		if (!(inputFeature.contains(feature) && inputFeature.contains(trueValue)))
		{
			inputFeature.add(feature);
			outputLabel.add(trueValue);
		}

		transit();
		evaluate(loss, inputFeature, outputLabel);
		determine();

		double[][][] weight = globalBestSolution.getWeight();
		double[][] bias = globalBestSolution.getBias();
		for (int i = 0; i < layers.length; i++)
		{
			layers[i].updateWeight(weight[i]);
			layers[i].updateBias(bias[i]);
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

	private void evaluate(AbstractLoss loss, ArrayList<double[]> feature, ArrayList<double[]> label)
	{
		// evaluate globalBestSolution loss with other feature input
		particle[particle.length - 1].setNowSolution(new Solution(globalBestSolution));
		particle[particle.length - 1].setLocalBestValue(0);

		for (int i = 0; i < particle.length; i++)
		{
			particle[i].setLocalBestValue(particle[i].getLocalBestValue() * 1.1);
		}

		for (int i = 0; i < particle.length; i++)
		{
			evaluate(particle[i], loss, feature, label);
		}
	}

	private void evaluate(Particle particle, AbstractLoss loss, ArrayList<double[]> feature, ArrayList<double[]> label)
	{
		double[][][] weight = particle.getNowSolution().getWeight();
		double[][] bias = particle.getNowSolution().getBias();
		double[] predictLabel;
		double error = 0;

		for (int i = 0; i < evaluateLayers.length; i++)
		{
			evaluateLayers[i].updateWeight(weight[i]);
			evaluateLayers[i].updateBias(bias[i]);
		}

		for (int i = 0; i < feature.size(); i++)
		{
			predictLabel = predict(feature.get(i));
			error = error + loss.getError(predictLabel, label.get(i));
		}

		if (error < particle.getLocalBestValue() || particle.getLocalBestValue() == 0)
		{
			particle.setLocalBestValue(error);
			particle.setLocalBestSolution(particle.getNowSolution());
		}
	}

	private void determine()
	{
		globalBestValue = globalBestValue * 1.1;
		for (int i = 0; i < particle.length; i++)
		{
			if (particle[i].getLocalBestValue() < globalBestValue || globalBestValue == 0)
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
}
