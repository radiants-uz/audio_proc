from sqladmin import ModelView

from app.models.document import Document


class DocumentAdmin(ModelView, model=Document):
    icon = "fa-solid fa-file-audio"
    can_create = False
    can_edit = False

    column_list = [
        Document.id, Document.name, Document.kind, Document.operation_name,
        Document.service_name, Document.language_code, Document.parent_id, Document.created_at,
    ]
    column_searchable_list = [Document.name, Document.object_key]
    column_sortable_list = [Document.id, Document.created_at]
    column_default_sort = [(Document.created_at, True)]
