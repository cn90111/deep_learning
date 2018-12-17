package optimizer;

import layer.Layer;
import loss.AbstractLoss;

public abstract class AbstractOptimizer
{
	public AbstractOptimizer()
	{
	}

	public abstract void setLayers(Layer[] layers);

	public abstract void update(AbstractLoss loss, double guessValue[], double trueValue[]);
}
