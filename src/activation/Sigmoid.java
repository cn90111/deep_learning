package activation;

public class Sigmoid extends AbstractActivation
{
	@Override
	protected double calculate(double data)
	{
		return 1 / (1 + Math.exp(-1 * data));
	}
}
