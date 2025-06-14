from flask import Flask, request, jsonify, send_file
import pandas as pd
from io import StringIO, BytesIO
from SignalAnalysis import SignalAnalysis
import json
import zipfile

app = Flask(__name__)
@app.route('/')
def index():
    return 'Flask app is running!'

@app.route('/upload', methods=['POST'])
def upload_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    print("Received file:", file.filename)
    print("Content-Length Header:", request.headers.get('Content-Length'))

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file.filename.endswith('.csv'):
        return jsonify({'error': 'Only CSV files are allowed'}), 400

    try:
        json_metadata = request.form.get('metadata')
        if json_metadata:
            json_metadata = json.loads(json_metadata)

        sa = SignalAnalysis()
        csv_data = StringIO(file.stream.read().decode('utf-8'))
        sa.load_data(csv_data)

        if json_metadata:
            if json_metadata["signal_conditioning"]["detrending"]:
                sa.detrend_data()

            if json_metadata["signal_conditioning"]["decimation"]:
                sa.decimate_data()

            if json_metadata["signal_conditioning"]["filter"]:
                sa.apply_filter(json_metadata["signal_conditioning"]["filter"])

            if json_metadata["fft"]:
                sa.fft(json_metadata["fft"])

            if json_metadata["damping"]["damping_order"]:
                sa.damping(json_metadata["damping"]["damping_order"])

            if json_metadata["auto_configuration"]:
                sa.autocorrelate()

            if json_metadata["cross_configuration"]:
                sa.cross_correlate()

        # Create a ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr('original_df.csv', sa.df.to_csv(index=False))
            zf.writestr('processed_df.csv', sa.processed_df.to_csv(index=False))
            zf.writestr('fft_df.csv', sa.fft_df.to_csv(index=False))
            zf.writestr('autocorr_df.csv', sa.autocorr_df.to_csv(index=False))

            # If crosscorr_df is a dictionary of DataFrames
            for (col1, col2), df in sa.crosscorr_df.items():
                filename = f"crosscorr_{col1}_{col2}.csv"
                zf.writestr(filename, df.to_csv(index=False))

        zip_buffer.seek(0)

        return send_file(
            zip_buffer,
            mimetype='application/zip',
            as_attachment=True,
            download_name='processed_data.zip'
        )


    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)