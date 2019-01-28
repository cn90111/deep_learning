package optimizer;

import layer.Layer;
import loss.AbstractLossFunction;

public abstract class Optimizer
{
	protected Layer[] layers;
	protected AbstractLossFunction lossFunction;
	protected int currentEpoch;

	public Optimizer()
	{
	}

	public abstract void setConfiguration(Layer[] layers, AbstractLossFunction lossFunction);

	public abstract void update(double guessValue[], double trueValue[]);

	public abstract void newEpoch(int currentEpoch);
}
