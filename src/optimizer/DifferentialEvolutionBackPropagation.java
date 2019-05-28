package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;

public class DifferentialEvolutionBackPropagation extends MetaheuristicOptimizer
{

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

}
