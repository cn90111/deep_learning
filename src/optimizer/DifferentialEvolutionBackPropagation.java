package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;

public class DifferentialEvolutionBackPropagation extends MetaheuristicOptimizer
{

	public DifferentialEvolutionBackPropagation(int dataSize)
	{
		super(dataSize);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction)
	{
		super.setConfiguration(layers, lossFunction);
	}

	@Override
	public void update(double[] guessValue, double[] trueValue)
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void newEpoch(int currentEpoch)
	{
		// TODO Auto-generated method stub
		
	}

	@Override
	public void update()
	{
		// TODO Auto-generated method stub
		
	}

}
