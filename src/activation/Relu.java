package activation;

public class Relu extends AbstractActivation
{
	@Override
	protected double calculate(double data)
	{
		return 0 > data ? 0 : data;
	}
	
	
}
