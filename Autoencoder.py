import torch
import torch.nn.functional as F
from MLP import MLP, get_batches, guardar_MLP, cargar_MLP, nombre_a_func






class Autoencoder:
    """
    Clase que representa un autoencoder.
    - encoder: red neuronal que codifica la entrada.
    - decoder: red neuronal que decodifica la representación comprimida.
    """
    def __init__(self, encoder : MLP, decoder : MLP):

        self.encoder = encoder
        self.decoder = decoder
        self.layers = encoder.layers + decoder.layers
        self.dim_latente = self.encoder.dim_out
        self.description = "Autoencoder"
        
        if self.encoder.description == "MLP":
            self.encoder.description = "Encoder"
        if self.decoder.description == "MLP":
            self.decoder.description = "Decoder"


            
    def add_descript(self, text:str):
        self.description += "\n " + text
    
    def view_descript(self):
        print(self.description)

    def parameters(self):
        return self.encoder.parameters() + self.decoder.parameters()
    
    def weights(self):
        """Devuelve solo los pesos (w) de encoder y decoder."""
        return self.encoder.weights() + self.decoder.weights()

    def __call__(self,x):
        """
        Propaga la entrada x a través del codificador y decodificador.
        - Primero pasa por el codificador para obtener una representación comprimida.
        - Luego pasa por el decodificador para reconstruir la entrada original.
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def train_model(self, training_data: list[torch.Tensor],
                    n_steps: int, step_sz: float,
                    loss_f: callable = F.mse_loss, batch_size: int = None,
                    beta_l1: float = None, beta_kl: float=None, lambda_l2: float = None):
        """
        Entrena el autoencoder con:
        - Penalización L1 sobre la capa latente (sparsity).
        - Regularización L2 de pesos.
        
        Parámetros:
        - beta: peso de la penalización L1 sobre la representación latente.
        - lambda_l2: peso de la regularización L2 sobre los pesos.
        - eps: valor pequeño para evitar divisiones por cero en cálculos.
        """
        
        # Valores si recibimos None
        batch_size = len(training_data) if batch_size is None else batch_size
        beta_l1 = 0.0 if beta_l1 is None else beta_l1
        beta_kl = 0.0 if beta_kl is None else beta_kl
        lambda_l2 = 0.0 if lambda_l2 is None else lambda_l2
        

        parameters = self.parameters()
        weights = self.weights()

        for k in range(n_steps):
            # Log de perdidad
            epoch_loss = 0.0
            epoch_recon = 0.0
            epoch_l1 = 0.0
            epoch_kl = 0.0
            epoch_l2 =0.0
            num_batches = 0

            for X_batch, Y_batch in get_batches(training_data, training_data, batch_size):

                # Tensores
                X_batch = torch.stack(X_batch)  # (B, dim_in)
                if loss_f.__name__ == "cross_entropy": #No deberia recibir esta funcion de perdidad
                    # Y_batch: índices de clase
                    Y_batch = torch.tensor(Y_batch, dtype=torch.long)
                else:
                    Y_batch = torch.stack(Y_batch)
                
                # Gradientes a 0
                for p in parameters:
                    if p.grad is not None:
                        p.grad.zero_()

                # --- Forward ---
                encoded_batch = self.encoder(X_batch)    # Capa latente
                decoded_batch = self.decoder(encoded_batch)  # Reconstrucción

                # Pérdida de reconstrucción
                loss_recon = loss_f(decoded_batch,Y_batch) 

                # Penalización L1 sobre capa latente
                if beta_l1 > 0.0:
                    loss_l1 = torch.mean(torch.abs(encoded_batch))
                else:
                    loss_l1 = 0.0

                # Sparsity por KL divergence
                if beta_kl > 0.0:
                    rho = 0.05  # sparsity objetivo
                    rho_hat = torch.clamp(torch.mean(encoded_batch, dim=0), 1e-6, 1-1e-6)

                    kl_div = rho * torch.log((rho + 1e-8) / (rho_hat + 1e-8)) +  (1 - rho) * torch.log((1 - rho + 1e-8) / (1 - rho_hat + 1e-8))

                    loss_kl = torch.sum(kl_div)  # sumar sobre todas las neuronas
                else:
                    loss_kl = 0.0

                # Regularización L2 sobre todos los parámetros
                if lambda_l2 > 0.0:
                    loss_l2 = sum(torch.sum(w**2) for w in weights) / X_batch.size(0)
                else:
                    loss_l2 = 0.0

                # Pérdida total
                loss = loss_recon + beta_l1 * loss_l1 + beta_kl * loss_kl + lambda_l2 * loss_l2

                # --- Backward ---
                loss.backward()

                # --- Actualización de parámetros ---
                for p in parameters:
                    p.data -= step_sz * p.grad
                
                # --- Actualizacion perdida(log) ---
                epoch_loss += loss.item()
                epoch_recon += loss_recon.item()
                epoch_l1 += loss_l1.item()
                epoch_kl += loss_kl.item()
                epoch_l2 += loss_l2.item()
                num_batches += 1

            # Log
            if k % 50 == 0 or k == n_steps - 1:
                avg_loss = epoch_loss / num_batches
                avg_recon = epoch_recon / num_batches
                avg_l1 = epoch_l1 / num_batches
                avg_kl = epoch_kl / num_batches
                avg_l2 = epoch_l2 / num_batches
                print(f"Paso {k} | Loss total: {avg_loss:.6f} "
                    f"(Recon: {avg_recon:.6f}, L1: {avg_l1:.6f}, KL: {avg_kl:.6f}, L2: {avg_l2:.6f})")
                

def guardar_autoencoder( red : Autoencoder, archivo : str):
    """
    Guarda el estado de un autoencoder en un archivo JSON.
    - autoencoder: objeto Autoencoder a guardar.
    - archivo: ruta del archivo destino.
    Se almacena:
      - estructura del codificador y decodificador
      - funciones de activación
      - pesos de ambos componentes
    """
    state = [p.detach().tolist() for p in red.parameters()]
    estructura_enc = red.encoder.dims
    estructura_dec = red.decoder.dims
    descripcion = red.description
    activaciones_enc = [ getattr(f_act, "__name__", "none") if f_act else "none" for f_act in red.encoder.activaciones]
    activaciones_dec = [ getattr(f_act, "__name__", "none") if f_act else "none" for f_act in red.decoder.activaciones]
    
    import json
    with open(archivo, "w") as f:
        json.dump({
            "tipo_modelo": "autoencoder",
            "descripcion": descripcion,
            "estructura_encoder": estructura_enc,
            "estructura_decoder": estructura_dec,
            "activaciones_encoder": activaciones_enc,
            "activaciones_decoder": activaciones_dec,
            "pesos": state
        }, f)
    print(f"Autoencoder guardado en '{archivo}'\n")

def cargar_autoencoder(archivo : str):
    """
    Carga un autoencoder desde un archivo JSON.
    - archivo: ruta del archivo JSON con la configuración del autoencoder.
    Devuelve un objeto Autoencoder con el codificador y decodificador cargados.
    """
    import json
    with open(archivo, "r") as f:
        data = json.load(f)
    
    act_encoder = [nombre_a_func.get(f_act,None) for f_act in data["activaciones_encoder"]]
    act_decoder = [nombre_a_func.get(f_act,None) for f_act in data["activaciones_decoder"]]
    descripcion = data["descripcion"]
    encoder = MLP(
        dim_in=data["estructura_encoder"][0],
        dim_out=data["estructura_encoder"][-1],
        estructura_oct=data["estructura_encoder"][1:-1],
        f_act_list= act_encoder
    )

    decoder = MLP(
        dim_in=data["estructura_decoder"][0],
        dim_out=data["estructura_decoder"][-1],
        estructura_oct=data["estructura_decoder"][1:-1],
        f_act_list= act_decoder
    )

    for p, w in zip(encoder.parameters() + decoder.parameters(), data["pesos"]):
        p.data = torch.tensor(w, dtype=torch.float32)
    
    autoencoder = Autoencoder(encoder, decoder)
    autoencoder.description = descripcion

    return autoencoder

