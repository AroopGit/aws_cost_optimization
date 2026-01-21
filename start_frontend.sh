#!/bin/bash
echo "Starting Frontend..."
echo ""
echo "Opening index.html in your default browser..."
echo ""
echo "If you prefer using a web server, run: python -m http.server 3000"
echo ""

# Try to open in default browser
if command -v xdg-open > /dev/null; then
    xdg-open index.html
elif command -v open > /dev/null; then
    open index.html
else
    echo "Please open index.html manually in your browser"
fi

