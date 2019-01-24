package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;

public abstract class AbstractOptimizer
{
	public AbstractOptimizer()
	{
	}

	public abstract void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction);

	public abstract void update(double guessValue[], double trueValue[]);
}
