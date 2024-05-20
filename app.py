import gradio as gr
from datetime import datetime

from code import (initialize_session, 
                  image_to_df,
                  update_combined_df,
                  visualize_all       
                  )

with gr.Blocks(gr.themes.Soft(primary_hue="emerald",
                              font=[gr.themes.GoogleFont("Quicksand"),"ui-sans-serif", "system-ui", "sans-serif"],
                              font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas","monospace"]
                              )) as demo:
    session_id = gr.State(initialize_session())

    gr.Markdown("# In this demo you can upload your receipt photo, make changes to data and visualize your expenses.")
    gr.Markdown("""
    ## Important Information
    - **Do not share your OpenAI API key**. Keep it secure and private.
    - Use environment-specific keys with limited permissions whenever possible.
    - The API key is used only during the session and is not stored permanently.
    """)

    with gr.Tabs():
        with gr.TabItem("Receipt info"):

            with gr.Row():
              api_key_input = gr.Textbox(label="Enter your OpenAI API key", type="password",max_lines=1,placeholder="sk-...")

            with gr.Row():
                with gr.Column(scale=1):
                  gr.Markdown("# Upload your receipts")
                with gr.Column(scale=2):
                  gr.Markdown("""
                  # Check and edit your data.
                  The model makes mistakes, for this reason, check extracted data, apply changes, then hit 'Confirm' button.""")

            with gr.Row():
                with gr.Column(scale=1):
                  image_input = gr.Files(type="filepath", file_count="multiple", scale=1)
                with gr.Column(scale=2):
                  gr.Markdown("""Possible categories:
                              'protein_foods', 'dairy', 'fruits', 'vegetables', 'grains', 'nuts_and_seeds','beverages', 'snacks',
                              'condiments', 'frozen_foods', 'bakery', 'canned_goods','household', 'personal_care', 'pet_supplies', 'other'""")




            with gr.Row():

                uploaded_images = gr.Gallery(label="Uploaded Receipt Images")
                image_output = gr.Dataframe(interactive=True,
                                            headers=['store', 'address', 'city', 'phone', 'receipt_no', 'date', 'time','total', 'number_items',
                                                     'payment_method', 'week', 'month', 'name','unit', 'price', 'amount', 'category'],
                                            col_count=(17, "fixed"),
                                            wrap=True,
                                            scale=2)

                combined_df = gr.Dataframe(visible=False)


            with gr.Row():
                with gr.Column(scale=1):
                    upload_button = gr.Button("Upload receipt")
                with gr.Column(scale=1):
                    combine_button = gr.Button("Confirm")
                with gr.Column(scale=1):
                    receipt_count_text = gr.Textbox(label="Uploaded files:", interactive=False)

            gr.Markdown("""
            # Visualize Expenses
            **Choose the base month and set your monthly budget, then hit 'Visualize'**""")

            with gr.Row():
                month = gr.Dropdown(value = datetime.now().strftime('%b'), 
                                    choices=["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"], 
                                    label="Month")
                
                budget = gr.Slider(0, 1000, 10, label="Budget")
            visualize_button = gr.Button("Visualize")

            with gr.Row():
                with gr.Column(scale=0.4):
                    fig1 = gr.Plot()
                with gr.Column(scale=0.6):
                    fig2 = gr.Plot()

            with gr.Row():
                with gr.Column(scale=0.4):
                    fig3 = gr.Plot()
                with gr.Column(scale=0.6):
                    fig4 = gr.Plot()

            with gr.Row():
                with gr.Column(scale=0.4):
                    fig5 = gr.Plot()
                with gr.Column(scale=0.6):
                    fig6 = gr.Plot()



    upload_button.click(image_to_df, inputs=[image_input, api_key_input], outputs=image_output)
    upload_button.click(fn=lambda files: [file.name for file in files], inputs=image_input, outputs=uploaded_images)
    combine_button.click(fn=update_combined_df, inputs=[image_output, session_id], outputs=[combined_df, receipt_count_text])
    visualize_button.click(fn=visualize_all, inputs=[combined_df, month, budget], outputs=[fig1, fig2, fig3, fig4, fig5, fig6])

demo.launch(debug=True)

