"""
Utility functions for the auto_post package.
Includes Portable Text conversion and title list management.
"""

import re
import json
import uuid

from .config import TITLES_FILE


def generate_key():
    """Generate a unique key for Portable Text blocks."""
    return uuid.uuid4().hex[:12]


def load_title_list():
    """Load pending titles from titles.json."""
    try:
        with open(TITLES_FILE, 'r') as f:
            data = json.load(f)
            return data.get('titles', [])
    except FileNotFoundError:
        print("Warning: titles.json not found")
        return []
    except json.JSONDecodeError:
        print("Warning: titles.json is invalid")
        return []


def save_title_list(titles):
    """Save updated titles list after removing used title."""
    try:
        with open(TITLES_FILE, 'w') as f:
            json.dump({'titles': titles}, f, indent=2)
        print(f"Updated titles.json ({len(titles)} titles remaining)")
        return True
    except Exception as e:
        print(f"Error saving titles.json: {e}")
        return False


def parse_inline_content(text):
    """Parse inline content (bold, italic) without links."""
    children = []
    parts = re.findall(r'\*\*(.+?)\*\*|\*([^*]+?)\*|([^*]+)', text)

    for bold, italic, normal in parts:
        if bold:
            children.append({
                "_type": "span",
                "_key": generate_key(),
                "text": bold,
                "marks": ["strong"]
            })
        elif italic:
            children.append({
                "_type": "span",
                "_key": generate_key(),
                "text": italic,
                "marks": ["em"]
            })
        elif normal:
            children.append({
                "_type": "span",
                "_key": generate_key(),
                "text": normal,
                "marks": []
            })

    if not children:
        children.append({
            "_type": "span",
            "_key": generate_key(),
            "text": text,
            "marks": []
        })

    return children


def parse_inline_with_links(text):
    """Parse inline content including links, bold, and italic."""
    children = []
    mark_defs = []

    link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    last_end = 0

    for match in re.finditer(link_pattern, text):
        if match.start() > last_end:
            before_text = text[last_end:match.start()]
            children.extend(parse_inline_content(before_text))

        link_key = generate_key()
        link_text = match.group(1)
        link_url = match.group(2)

        mark_defs.append({
            "_type": "link",
            "_key": link_key,
            "href": link_url
        })

        children.append({
            "_type": "span",
            "_key": generate_key(),
            "text": link_text,
            "marks": [link_key]
        })

        last_end = match.end()

    if last_end < len(text):
        remaining_text = text[last_end:]
        children.extend(parse_inline_content(remaining_text))

    if not children:
        children.append({
            "_type": "span",
            "_key": generate_key(),
            "text": text,
            "marks": []
        })

    return children, mark_defs


def convert_markdown_to_portable_text(markdown_content):
    """
    Convert Markdown content to Sanity Portable Text format.
    Handles paragraphs, headings, bold, italic, links, and lists.
    """
    blocks = []
    lines = markdown_content.split('\n')
    current_list_type = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip empty lines
        if not line:
            current_list_type = None
            i += 1
            continue

        # Check for headings
        if line.startswith('### '):
            current_list_type = None
            blocks.append({
                "_type": "block",
                "_key": generate_key(),
                "style": "h3",
                "markDefs": [],
                "children": parse_inline_content(line[4:])
            })
            i += 1
            continue

        if line.startswith('## '):
            current_list_type = None
            blocks.append({
                "_type": "block",
                "_key": generate_key(),
                "style": "h2",
                "markDefs": [],
                "children": parse_inline_content(line[3:])
            })
            i += 1
            continue

        # Check for bullet lists
        if line.startswith('- ') or line.startswith('* '):
            current_list_type = 'bullet'
            list_item_content = line[2:]
            children, mark_defs = parse_inline_with_links(list_item_content)
            blocks.append({
                "_type": "block",
                "_key": generate_key(),
                "style": "normal",
                "listItem": "bullet",
                "level": 1,
                "markDefs": mark_defs,
                "children": children
            })
            i += 1
            continue

        # Check for numbered lists
        numbered_match = re.match(r'^(\d+)\.\s+(.+)$', line)
        if numbered_match:
            current_list_type = 'number'
            list_item_content = numbered_match.group(2)
            children, mark_defs = parse_inline_with_links(list_item_content)
            blocks.append({
                "_type": "block",
                "_key": generate_key(),
                "style": "normal",
                "listItem": "number",
                "level": 1,
                "markDefs": mark_defs,
                "children": children
            })
            i += 1
            continue

        # Regular paragraph
        current_list_type = None
        children, mark_defs = parse_inline_with_links(line)
        blocks.append({
            "_type": "block",
            "_key": generate_key(),
            "style": "normal",
            "markDefs": mark_defs,
            "children": children
        })
        i += 1

    return blocks
