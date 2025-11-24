# Backend del proyecto aplicacion

Utiliza el puerto 3007

**Importante**

Usa el venv.

```
uv venv
<activa venv segun tu terminal>
```

Primero, instala todas las dependencias

```
uv install
```

Esto va a tardar.

Luego, entrena el modelo. Esto no deberia tardar tanto.
```
uv run train.py
```

Y entonces recien levantas el servidor.

```
uv run main.py
```
