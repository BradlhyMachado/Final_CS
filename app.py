from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

classifier = pipeline('sentiment-analysis',
                      model="nlptown/bert-base-multilingual-uncased-sentiment")

@app.route('/analizar_sentimiento', methods=['POST'])
def analizar_sentimiento():
    try:
        data = request.get_json()
        comentario = data['comentario']
        resultado = classifier(comentario)[0]
        
        polaridad = resultado['label']
        score = round(resultado['score'], 4)
        
        return jsonify({'polaridad': polaridad, 'score': score})

    except Exception as e:
        return jsonify({'error': str(e)})


@app.route('/promedio_sentimientos', methods=['POST'])
def analizar_sentimiento():
    try:
        data = request.get_json()
        comentarios = data['comentarios']

        total_polaridad = 0
        total_comentarios = len(comentarios)

        for comentario in comentarios:
            resultado = classifier(comentario)[0]
            
            total_polaridad += int(resultado['label'][0])

        if total_comentarios > 0:
            promedio_polaridad = round(total_polaridad / total_comentarios, 2)
        else:
            promedio_polaridad = 0

        return jsonify({'promedio_polaridad': promedio_polaridad})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)