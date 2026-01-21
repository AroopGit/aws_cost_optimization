import React, { useState } from 'react';

const CsvProcessor = () => {
    const [file, setFile] = useState(null);
    const [processing, setProcessing] = useState(false);
    const [error, setError] = useState(null);

    const handleFileChange = (e) => {
        if (e.target.files) {
            setFile(e.target.files[0]);
            setError(null);
        }
    };

    const handleProcess = async () => {
        if (!file) return;
        setProcessing(true);
        setError(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('http://localhost:8000/pricing/process_csv', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `processed_${file.name}`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
            } else {
                const errData = await response.json();
                setError(errData.error || 'Failed to process CSV');
            }
        } catch (error) {
            console.error('Error:', error);
            setError('Network error or backend unavailable');
        } finally {
            setProcessing(false);
        }
    };

    return (
        <div className="rounded-2xl border border-slate-200 bg-white p-6 shadow-sm mt-6">
            <div className="flex items-center justify-between mb-4">
                <div>
                    <h3 className="text-lg font-semibold text-slate-900">Batch Processing</h3>
                    <p className="text-sm text-slate-500">Upload a CSV file to generate pricing recommendations</p>
                </div>
            </div>

            <div className="flex flex-col gap-4">
                <div className="flex items-center gap-4">
                    <input
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                        className="block w-full text-sm text-slate-500
              file:mr-4 file:py-2 file:px-4
              file:rounded-full file:border-0
              file:text-sm file:font-semibold
              file:bg-blue-50 file:text-blue-700
              hover:file:bg-blue-100
              cursor-pointer
            "
                    />
                    <button
                        onClick={handleProcess}
                        disabled={!file || processing}
                        className={`px-6 py-2 rounded-lg text-white font-medium transition-all shadow-sm flex-shrink-0 ${!file || processing
                            ? 'bg-slate-400 cursor-not-allowed'
                            : 'bg-blue-600 hover:bg-blue-700 hover:shadow-md active:transform active:scale-95'
                            }`}
                    >
                        {processing ? (
                            <span className="flex items-center gap-2">
                                <svg className="animate-spin h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                </svg>
                                Processing...
                            </span>
                        ) : 'Process & Download'}
                    </button>
                </div>

                {error && (
                    <div className="p-3 rounded-lg bg-red-50 text-red-600 text-sm border border-red-100">
                        {error}
                    </div>
                )}

                <div className="text-xs text-slate-400">
                    Required columns: <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-600">sku_id</code>, <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-600">channel</code>, <code className="bg-slate-100 px-1 py-0.5 rounded text-slate-600">region</code>
                </div>
            </div>
        </div>
    );
};

export default CsvProcessor;
