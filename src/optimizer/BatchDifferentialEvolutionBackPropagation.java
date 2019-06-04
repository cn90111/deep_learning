package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;
import metaheuristic.DeParameter;
import metaheuristic.DeSolution;

public class BatchDifferentialEvolutionBackPropagation extends DifferentialEvolutionBackPropagation
		implements SupportBatchUpdate
{

	public BatchDifferentialEvolutionBackPropagation(DeParameter de, int batch, int deGenerations, double learningRate,
			double learningRateDecayRate)
	{
		super(de, batch, deGenerations, learningRate, learningRateDecayRate);
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);

		resetBatch();
	}

	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		super.update(guessValue, trueValue);
	}

	@Override
	public void update()
	{
		batchUpdate();
	}

	@Override
	public void batchUpdate()
	{
		super.update();
		resetBatch();
	}

	protected void evaluate(double[][] feature, double[][] label)
	{
		setSolutionWeightToLayers(globalBestSolution);
		globalBestValue = 0;
		for (int i = 0; i < feature.length; i++)
		{
			globalBestValue = globalBestValue + evaluate(feature[i], label[i]);
		}

		for (int i = 0; i < solutions.length; i++)
		{
			evaluate(solutions[i], feature, label);
		}
	}

	protected void evaluate(DeSolution solution, double[][] feature, double[][] label)
	{
		double lossValue = 0;
		setSolutionWeightToLayers(solution.getNewSolution());
		for (int i = 0; i < feature.length; i++)
		{
			lossValue = lossValue + evaluate(feature[i], label[i]);
		}
		solution.setNewValue(lossValue);

		lossValue = 0;
		setSolutionWeightToLayers(solution.getNowSolution());
		for (int i = 0; i < feature.length; i++)
		{
			lossValue = lossValue + evaluate(feature[i], label[i]);
		}
		solution.setNowValue(lossValue);
	}

	@Override
	public int getBatchSize()
	{
		return dataSize;
	}

	@Override
	public void resetBatch()
	{
		dataCount = 0;

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
		firstEvalutate = true;
	}
}
