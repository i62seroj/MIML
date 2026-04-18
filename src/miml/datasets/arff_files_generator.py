import random
import os

def generate_miml_arff(
    output_path,
    n_bags=100,
    min_instances=3,
    max_instances=10,
    n_features=5,
    n_labels=4,
    label_density=0.3
):
    """
    Genera un archivo ARFF estilo MIML.

    Parámetros:
    - n_bags: número de bags (filas)
    - min_instances, max_instances: rango de instancias por bag
    - n_features: atributos por instancia
    - n_labels: número de etiquetas
    - label_density: probabilidad de que una etiqueta sea 1
    """

    with open(output_path, "w") as f:
        # Cabecera
        f.write("@RELATION miml_generated\n\n")

        # Atributo relacional (instancias dentro de cada bag)
        f.write("@ATTRIBUTE bag relational\n")
        for i in range(n_features):
            f.write(f"    @ATTRIBUTE f{i+1} NUMERIC\n")
        f.write("@END bag\n\n")

        # Etiquetas
        for i in range(n_labels):
            f.write(f"@ATTRIBUTE label{i+1} {{0,1}}\n")

        f.write("\n@DATA\n")

        # Datos
        for _ in range(n_bags):
            n_instances = random.randint(min_instances, max_instances)

            # Generar instancias
            instances = []
            for _ in range(n_instances):
                features = [round(random.uniform(0, 1), 4) for _ in range(n_features)]
                instances.append(",".join(map(str, features)))

            # ARFF usa \n dentro del string
            bag_str = "\\n".join(instances)

            # Generar etiquetas
            labels = [
                "1" if random.random() < label_density else "0"
                for _ in range(n_labels)
            ]

            # Escribir línea
            f.write(f"\"{bag_str}\",{','.join(labels)}\n")


# ====== USO ======
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(script_dir, "miml_large.arff")

    generate_miml_arff(
        output_path=output_file,
        n_bags=1000,          # 🔥 número de filas
        min_instances=5,
        max_instances=20,
        n_features=10,
        n_labels=6,
        label_density=0.4
    )

    print(f"Archivo generado en: {output_file}")