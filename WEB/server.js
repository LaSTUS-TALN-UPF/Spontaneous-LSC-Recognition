const express = require('express');
const path = require('path');
const fs = require('fs');

const app = express();
const videoFolder = path.join(__dirname, 'videos'); // Carpeta donde están los videos
const poseVideoFolder = path.join(__dirname, 'videos_pose');

// Middleware para servir archivos estáticos
app.use(express.static('public'));
app.use('/videos', express.static(videoFolder)); // Servir archivos de la carpeta "videos"
app.use('/videos_pose', express.static(poseVideoFolder)); 
// Ruta para obtener los nombres únicos de los videos
app.get('/videos', (req, res) => {
    fs.readdir(videoFolder, (err, files) => {
        if (err) {
            return res.status(500).send('Error al leer los videos');
        }

        // Extraer solo los nombres únicos antes del primer "_"
        const videoNamesSet = new Set();
        const videoMap = {}; // Guardará la lista de archivos para cada nombre

        files.forEach(file => {
            const name = file.split('_')[0]; // Tomamos solo lo antes del "_"
            videoNamesSet.add(name);

            if (!videoMap[name]) {
                videoMap[name] = [];
            }
            videoMap[name].push(file); // Guardamos todos los videos con ese nombre
        });

        // Convertir en un array de objetos con nombre y lista de archivos
        const videoNames = Array.from(videoNamesSet).map(name => ({
            name,
            videos: videoMap[name]
        }));

        res.json(videoNames);
    });
});

const port = 3000;
app.listen(port, () => {
    console.log(`Servidor corriendo en http://localhost:${port}`);
});
