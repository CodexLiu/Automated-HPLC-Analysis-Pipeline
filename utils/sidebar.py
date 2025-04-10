import dash
from dash import callback_context

def toggle_sidebar(n_clicks, current_state, current_button_style):
    ctx = callback_context
    is_initial_load = not ctx.triggered or ctx.triggered_id == '.' # Check if it's initial load

    # Determine the *target* collapsed state
    if is_initial_load:
        # On initial load, use the state from the store
        collapsed = current_state.get('collapsed', False) # Default to expanded if not set
        new_state = dash.no_update # Don't update the store itself on initial load
    else:
        # On button click, toggle the state
        collapsed = not current_state.get('collapsed', False)
        new_state = {'collapsed': collapsed}

    # --- Define Styles ---
    expanded_sidebar_style = {
        'width': '25%', 
        'float': 'left', 
        'padding': '10px', 
        'boxSizing': 'border-box', 
        'position': 'relative', # Keep relative for button positioning
        'transition': 'width 0.3s ease, padding 0.3s ease',
        'overflowY': 'auto', # Allow vertical scroll if content overflows
        'height': 'calc(100vh - 90px)' # Adjust height based on header/tabs
    }
    expanded_plot_style = {
        'width': '75%', 
        'float': 'right', 
        'padding': '10px', 
        'boxSizing': 'border-box', 
        'transition': 'width 0.3s ease'
    }
    collapsed_sidebar_style = {
        'width': '0px', 
        'padding': '0px', 
        'overflow': 'hidden', 
        'position': 'relative', # Keep relative for button positioning
        'transition': 'width 0.3s ease, padding 0.3s ease',
        'borderRight': 'none', # Hide border when collapsed
        'float': 'left', # Keep float for layout consistency
        'height': 'calc(100vh - 90px)' # Match height
    }
    collapsed_plot_style = {
        'width': '100%', 
        'padding': '10px', 
        'boxSizing': 'border-box', 
        'transition': 'width 0.3s ease',
        'float': 'none' # Allow plot to take full width
    }

    # Calculate styles and button text based on the target state
    if collapsed:
        sidebar_style = collapsed_sidebar_style
        plot_style = collapsed_plot_style
        button_text = ">" # Show > when collapsed
    else:
        sidebar_style = expanded_sidebar_style
        plot_style = expanded_plot_style
        button_text = "<" # Show < when expanded

    # Calculate button style
    # Use provided style or initialize if it's None (first load)
    button_style = current_button_style.copy() if current_button_style else {
        'position': 'absolute', 'top': '10px', 'zIndex': '10',
        'padding': '5px 8px', 'fontSize': '12px', 'cursor': 'pointer',
        'border': '1px solid #ccc', 'borderRadius': '0 4px 4px 0',
        'transition': 'left 0.3s ease'
    }
    if collapsed:
        button_style['left'] = '5px' # Position near left edge when collapsed
    else:
        # Adjust position based on the expanded sidebar width
        button_style['left'] = f"calc({expanded_sidebar_style['width']} - 30px)" 

    return new_state, sidebar_style, plot_style, button_text, button_style