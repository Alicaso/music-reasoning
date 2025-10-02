#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to add args_schema and returns_schema to all tools in music21_tools.yaml
"""

import yaml
import json

def create_args_schema(tool_name, arguments):
    """Create args_schema based on tool arguments"""
    required = []
    properties = {}
    
    for arg in arguments:
        arg_name = arg['name']
        arg_type = arg['type']
        is_optional = arg.get('is_optional', False)
        
        if not is_optional:
            required.append(arg_name)
        
        prop_schema = {"type": arg_type, "description": arg['description']}
        
        # Add constraints based on type and name
        if arg_type == 'integer':
            # measure numbers are 1-based
            if 'measure' in arg_name:
                prop_schema['minimum'] = 1
            # indices (part_index, event_index, generic index) are 0-based
            elif 'index' in arg_name or 'event' in arg_name:
                prop_schema['minimum'] = 0
        elif arg_type == 'string':
            if arg_name == 'kern_data':
                prop_schema['minLength'] = 1
            elif arg_name == 'key_string':
                prop_schema['minLength'] = 1
                prop_schema['pattern'] = r'^[A-Ga-g][#b]?[m]?$'
        
        properties[arg_name] = prop_schema
    
    return {
        "type": "object",
        "required": required,
        "properties": properties
    }

def normalize_args_schema(schema):
    """Ensure measure numbers are 1-based minimum in existing schemas."""
    try:
        props = schema.get('properties', {}) or {}
        for name, prop in props.items():
            if isinstance(prop, dict) and prop.get('type') == 'integer':
                if 'measure' in name:
                    prop['minimum'] = 1
        schema['properties'] = props
    except Exception:
        pass
    return schema

def create_returns_schema():
    """Create standardized returns_schema"""
    return {
        "type": "object",
        "required": ["ok", "data"],
        "properties": {
            "ok": {
                "type": "boolean",
                "description": "Whether the operation was successful."
            },
            "data": {
                "type": "string", 
                "description": "The result data or error message."
            }
        }
    }

def main():
    # Load the YAML file
    with open('music21_tools.yaml', 'r', encoding='utf-8') as f:
        tools = yaml.safe_load(f)
    
    # Process each tool (add only if missing; normalize constraints even if present)
    for tool in tools:
        tool_name = tool['tool_name']
        arguments = tool.get('arguments', [])

        if 'args_schema' not in tool or not tool.get('args_schema'):
            tool['args_schema'] = create_args_schema(tool_name, arguments)
        # Normalize existing/new args_schema
        tool['args_schema'] = normalize_args_schema(tool['args_schema'])
        if 'returns_schema' not in tool or not tool.get('returns_schema'):
            tool['returns_schema'] = create_returns_schema()
    
    # Write back to file
    with open('music21_tools.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(tools, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print("Successfully added schemas to all tools!")

if __name__ == "__main__":
    main()
