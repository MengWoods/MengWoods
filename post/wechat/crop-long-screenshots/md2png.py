import markdown
from playwright.sync_api import sync_playwright
import tempfile
import os

def markdown_to_html(md_content, base_path=""):
    """
    Convert Markdown content to HTML and style it to mimic the knb.im theme.
    """
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Markdown Preview</title>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 15px;
                line-height: 1.8;
                background-color: #f8f9fa;
                color: #333;
                max-width: 800px;
                margin: auto;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
                border-radius: 8px;
            }}
            h1 {{
                font-size: 2.5em;
                font-weight: 700;
                color: #1abc9c; /* Custom color for H1 */
                margin-bottom: 0.5em;
            }}
            h2 {{
                font-size: 2em;
                font-weight: 600;
                margin-bottom: 0.75em;
                color: #e74c3c; /* Custom color for H2 */
            }}
            h3 {{
                font-size: 1.5em;
                font-weight: 600;
                margin-bottom: 0.5em;
                color: #3498db; /* Custom color for H3 */
            }}
            p {{
                margin-bottom: 1.2em;
                color: #4a4a4a;
            }}
            blockquote {{
                margin: 1.5em 0;
                padding: 10px 20px;
                background: #f2f2f2;
                border-left: 5px solid #2c3e50;
                color: #555;
                font-style: italic;
            }}
            code {{
                font-family: 'Courier New', Courier, monospace;
                background: #f3f4f6;
                padding: 2px 6px;
                border-radius: 5px;
                color: #e63946;
                font-size: 0.95em;
            }}
            pre {{
                background: #272a36;
                color: #fff;
                padding: 15px;
                border-radius: 10px;
                overflow-x: auto;
                font-size: 0.9em;
            }}
            a {{
                color: #007bff;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 1.5em 0;
            }}
            table th, table td {{
                padding: 10px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            table th {{
                background-color: #f5f5f5;
                font-weight: bold;
            }}
            img {{
                max-width: 100%;
                height: auto;
                border-radius: 5px;
                margin: 1em 0;
            }}
            ul, ol {{
                margin: 1em 0;
                padding-left: 1.5em;
            }}
            strong {{
                color: #e67e22; /* Custom color for bold text */
                font-weight: bold;
            }}
        </style>


    </head>
    <body>
        {content}
    </body>
    </html>
    """
    # Convert Markdown to HTML
    html_content = markdown.markdown(md_content, extensions=["fenced_code", "tables"])

    # Replace relative image paths with absolute paths
    if base_path:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, "html.parser")
        for img in soup.find_all("img"):
            src = img.get("src")
            if src and not src.startswith(("http://", "https://", "file://")):
                abs_path = os.path.abspath(os.path.join(base_path, src))
                img["src"] = f"file://{abs_path}"
        html_content = str(soup)

    return html_template.format(content=html_content)


def save_html_to_file(html_content):
    """
    Save HTML content to a temporary file.
    """
    temp_dir = tempfile.gettempdir()
    file_path = os.path.join(temp_dir, "markdown_preview.html")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return file_path

def capture_screenshot(html_file_path, output_image_path, width=800):
    """
    Capture a screenshot of the HTML file using Playwright.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(device_scale_factor=2)  # 设置缩放倍率
        page = context.new_page()
        page.set_viewport_size({"width": width, "height": 800})  # 自定义宽度
        page.goto(f"file://{html_file_path}")
        page.wait_for_timeout(1000)  # 等待渲染完成
        page.screenshot(path=output_image_path, full_page=True)
        browser.close()


def markdown_to_screenshot(md_file_path, output_image_path, width=800):
    """
    Convert a Markdown file to a screenshot.
    """
    with open(md_file_path, "r", encoding="utf-8") as f:
        md_content = f.read()
    base_path = os.path.dirname(md_file_path)
    html_content = markdown_to_html(md_content, base_path=base_path)
    html_file_path = save_html_to_file(html_content)
    capture_screenshot(html_file_path, output_image_path, width)
    print(f"Screenshot saved to {output_image_path}")

# Example usage
if __name__ == "__main__":
    markdown_file = "/home/m/Documents/github/blog-resources/content/post/wechat/2024-11-20-same-as-ever/index.cn.md"  # Replace with your Markdown file path
    output_image = "screenshot.png"
    custom_width = 400  # Set custom width
    markdown_to_screenshot(markdown_file, output_image, width=custom_width)
