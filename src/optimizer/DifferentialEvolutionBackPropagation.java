package optimizer;

import activation.AbstractActivation;
import activation.Differentiable;
import layer.Layer;
import loss.AbstractLossFunction;
import metaheuristic.DeParameter;
import metaheuristic.DeSolution;
import metaheuristic.Solution;

public class DifferentialEvolutionBackPropagation extends MetaheuristicOptimizer
{
	private DeParameter de;

	private int deGenerations;
	private int deCount;

	private double learningRate;
	private double originLearningRate;
	private double learningRateDecayRate;

	// layer neurons weight
	private double[][][] weightDeltaArray;
	// layer bias
	private double[][] biasDeltaArray;

	private DeSolution[] solutions;

	private boolean firstEvalutate;

	// DE/best/1/bin strategy
	private int updateMode;// best or random
	private int groupNumber;// 1,2,3....

	public DifferentialEvolutionBackPropagation(DeParameter de, int dataSize, int deGenerations, double learningRate,
			double learningRateDecayRate)
	{
		super(dataSize);
		this.deGenerations = deGenerations;
		this.deCount = 0;
		this.originLearningRate = learningRate;
		this.learningRateDecayRate = learningRateDecayRate;
		this.firstEvalutate = true;

		for (int i = 0; i < de.size; i++)
		{
			solutions[i] = new DeSolution(de.solutionLimit);
		}
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);

		for (int i = 0; i < solutions.length; i++)
		{
			solutionsInit(solutions[i], globalBestSolution.getWeight(), globalBestSolution.getBias());
		}

	}

	private void solutionsInit(DeSolution solution, double[][][] weightSize, double[][] biasSize)
	{
		setSolution(solution, weightSize, biasSize);
	}

	protected void setSolution(DeSolution solution, double[][][] weight, double[][] bias)
	{
		double[][][] solutionWeight = null;
		double[][] solutionBias = null;
		solutionWeight = randomSetValueTo3DArray(weight, de.solutionLimit, -1 * de.solutionLimit);
		solutionBias = randomSetValueTo2DArray(bias, de.solutionLimit, -1 * de.solutionLimit);
		solution.setNewSolution(new Solution(solutionWeight, solutionBias));
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
		if (firstEvalutate == true)
		{
			evaluate(featureArray, labelArray);
			determine();
			firstEvalutate = false;
		}
		mutation();
		crossover();
		evaluate(featureArray, labelArray);
		determine();
	}

	private void evaluate(double[][] feature, double[][] label)
	{
		if (globalBestValue == 0)
		{
			setSolutionWeightToLayers(globalBestSolution);
			for (int i = 0; i < feature.length; i++)
			{
				globalBestValue = globalBestValue + evaluate(feature[i], label[i]);
			}
		}
		for (int i = 0; i < solutions.length; i++)
		{
			evaluate(solutions[i], feature, label);
		}
	}

	private void evaluate(DeSolution solution, double[][] feature, double[][] label)
	{
		double lossValue = 0;
		setSolutionWeightToLayers(solution.getNewSolution());
		for (int i = 0; i < feature.length; i++)
		{
			lossValue = lossValue + evaluate(feature[i], label[i]);
		}
		solution.setNewValue(lossValue);
	}

	private void determine()
	{
		DeSolution temp;
		for (int i = 0; i < solutions.length; i++)
		{
			temp = solutions[i];
			if (temp.getNewValue() < temp.getNowValue() || temp.getNowValue() == 0)
			{
				temp.setNowValue(temp.getNewValue());
				temp.setNowSolution(temp.getNewSolution());
			}

			if (temp.getNowValue() < globalBestValue)
			{
				globalBestValue = temp.getNowValue();
				globalBestSolution = temp.getNowSolution();
			}
		}
	}

	private void mutation()
	{
		for (int i = 0; i < solutions.length; i++)
		{
			int[] indexArray = randomArray(i);

			DeSolution[] randomSolutions = new DeSolution[3];

			for (int j = 0; j < 3; i++)
			{
				randomSolutions[j] = solutions[indexArray[j + 1]];
			}

			mutation(solutions[indexArray[0]], randomSolutions);
		}
	}

	private void mutation(DeSolution targetSolution, DeSolution[] referenceSolutions)
	{
		double[][][][] weight = new double[1 + referenceSolutions.length][][][];
		double[][][] bias = new double[1 + referenceSolutions.length][][];

		weight[0] = targetSolution.getNowSolution().getWeight();
		bias[0] = targetSolution.getNowSolution().getBias();
		for (int i = 0; i < weight.length; i++)
		{
			weight[i + 1] = referenceSolutions[i].getNowSolution().getWeight();
			bias[i + 1] = referenceSolutions[i].getNowSolution().getBias();
		}

		for (int i = 0; i < weight[0].length; i++)
		{
			for (int j = 0; j < weight[0][i].length; j++)
			{
				for (int k = 0; k < weight[0][i][j].length; k++)
				{
					weight[0][i][j][k] = weight[1][i][j][k] + de.f * (weight[2][i][j][k] - weight[3][i][j][k]);
				}
			}
		}

		for (int i = 0; i < bias[0].length; i++)
		{
			for (int j = 0; j < bias[0][i].length; j++)
			{
				bias[0][i][j] = bias[1][i][j] + de.f * (bias[2][i][j] - bias[3][i][j]);
			}
		}
		targetSolution.setNewSolution(new Solution(weight[0], bias[0]));
	}

	private void crossover()
	{
		for (int i = 0; i < solutions.length; i++)
		{
			crossover(solutions[i]);
		}
	}

	private void crossover(DeSolution solution)
	{
		double[][][][] weight = new double[2][][][];
		double[][][] bias = new double[2][][];

		weight[0] = solution.getNowSolution().getWeight();
		bias[0] = solution.getNowSolution().getBias();
		weight[1] = solution.getNewSolution().getWeight();
		bias[1] = solution.getNewSolution().getBias();

		for (int i = 0; i < weight[0].length; i++)
		{
			for (int j = 0; j < weight[0][i].length; j++)
			{
				for (int k = 0; k < weight[0][i][j].length; k++)
				{
					if (Math.random() < de.crossoverRate)
					{
						weight[1][i][j][k] = weight[1][i][j][k];
					}
					else
					{
						weight[1][i][j][k] = weight[0][i][j][k];
					}
				}
			}
		}

		for (int i = 0; i < bias[0].length; i++)
		{
			for (int j = 0; j < bias[0][i].length; j++)
			{
				if (Math.random() < de.crossoverRate)
				{
					weight[1][i][j] = weight[1][i][j];
				}
				else
				{
					weight[1][i][j] = weight[0][i][j];
				}
			}
		}
		solution.setNewSolution(new Solution(weight[1], bias[1]));
	}

	private int[] randomArray(int lockFirstIndexNumber)
	{
		int[] indexArray = new int[solutions.length];
		int randomNumber;
		int temp;

		for (int i = 0; i < solutions.length; i++)
		{
			indexArray[i] = i;
		}

		for (int i = 0; i < solutions.length; i++)
		{
			randomNumber = (int) (Math.random() * indexArray.length);
			temp = indexArray[i];
			indexArray[i] = indexArray[randomNumber];
			indexArray[randomNumber] = temp;
		}
		randomNumber = lockFirstIndexNumber;
		temp = indexArray[0];
		indexArray[0] = indexArray[randomNumber];
		indexArray[randomNumber] = temp;

		return indexArray;
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

		double[][][] weight = new double[evaluateLayers.length][][];
		double[][] bias = new double[evaluateLayers.length][];

		for (int i = 0; i < evaluateLayers.length; i++)
		{
			weight[i] = evaluateLayers[i].getWeight();
			bias[i] = evaluateLayers[i].getBias();
		}

		globalBestSolution.setWeight(weight);
		globalBestSolution.setBias(bias);

		for (int i = 0; i < layers.length; i++)
		{
			layers[i].updateWeight(weight[i]);
			layers[i].updateBias(bias[i]);
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
