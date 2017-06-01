
public class Main {

	public static void main(String[] args) {
		
		//ver os valores de treino, alpha e beta
		RBF_Network rbf = new RBF_Network(500, 3,5, 1, 150, 1.0, 0.7);
		
		rbf.Busca();

	}

}
