/**
 */
package promela.provider;


import java.util.Collection;
import java.util.List;

import org.eclipse.emf.common.notify.AdapterFactory;
import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EStructuralFeature;

import org.eclipse.emf.edit.provider.ComposeableAdapterFactory;
import org.eclipse.emf.edit.provider.IEditingDomainItemProvider;
import org.eclipse.emf.edit.provider.IItemLabelProvider;
import org.eclipse.emf.edit.provider.IItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.IItemPropertySource;
import org.eclipse.emf.edit.provider.IStructuredItemContentProvider;
import org.eclipse.emf.edit.provider.ITreeItemContentProvider;
import org.eclipse.emf.edit.provider.ItemPropertyDescriptor;
import org.eclipse.emf.edit.provider.ViewerNotification;

import promela.PromelaFactory;
import promela.PromelaPackage;
import promela.run_expr;

/**
 * This is the item provider adapter for a {@link promela.run_expr} object.
 * <!-- begin-user-doc -->
 * <!-- end-user-doc -->
 * @generated
 */
public class run_exprItemProvider
	extends any_exprItemProvider
	implements
		IEditingDomainItemProvider,
		IStructuredItemContentProvider,
		ITreeItemContentProvider,
		IItemLabelProvider,
		IItemPropertySource {
	/**
	 * This constructs an instance from a factory and a notifier.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public run_exprItemProvider(AdapterFactory adapterFactory) {
		super(adapterFactory);
	}

	/**
	 * This returns the property descriptors for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public List<IItemPropertyDescriptor> getPropertyDescriptors(Object object) {
		if (itemPropertyDescriptors == null) {
			super.getPropertyDescriptors(object);

			addProcPropertyDescriptor(object);
			addPriorityPropertyDescriptor(object);
		}
		return itemPropertyDescriptors;
	}

	/**
	 * This adds a property descriptor for the Proc feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addProcPropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_run_expr_proc_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_run_expr_proc_feature", "_UI_run_expr_type"),
				 PromelaPackage.Literals.RUN_EXPR__PROC,
				 true,
				 false,
				 true,
				 null,
				 null,
				 null));
	}

	/**
	 * This adds a property descriptor for the Priority feature.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected void addPriorityPropertyDescriptor(Object object) {
		itemPropertyDescriptors.add
			(createItemPropertyDescriptor
				(((ComposeableAdapterFactory)adapterFactory).getRootAdapterFactory(),
				 getResourceLocator(),
				 getString("_UI_run_expr_priority_feature"),
				 getString("_UI_PropertyDescriptor_description", "_UI_run_expr_priority_feature", "_UI_run_expr_type"),
				 PromelaPackage.Literals.RUN_EXPR__PRIORITY,
				 true,
				 false,
				 false,
				 ItemPropertyDescriptor.INTEGRAL_VALUE_IMAGE,
				 null,
				 null));
	}

	/**
	 * This specifies how to implement {@link #getChildren} and is used to deduce an appropriate feature for an
	 * {@link org.eclipse.emf.edit.command.AddCommand}, {@link org.eclipse.emf.edit.command.RemoveCommand} or
	 * {@link org.eclipse.emf.edit.command.MoveCommand} in {@link #createCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Collection<? extends EStructuralFeature> getChildrenFeatures(Object object) {
		if (childrenFeatures == null) {
			super.getChildrenFeatures(object);
			childrenFeatures.add(PromelaPackage.Literals.RUN_EXPR__ARG_LST);
		}
		return childrenFeatures;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EStructuralFeature getChildFeature(Object object, Object child) {
		// Check the type of the specified child object and return the proper feature to use for
		// adding (see {@link AddCommand}) it as a child.

		return super.getChildFeature(object, child);
	}

	/**
	 * This returns run_expr.gif.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object getImage(Object object) {
		return overlayImage(object, getResourceLocator().getImage("full/obj16/run_expr"));
	}

	/**
	 * This returns the label text for the adapted class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getText(Object object) {
		String label = ((run_expr)object).getLabel();
		return label == null || label.length() == 0 ?
			getString("_UI_run_expr_type") :
			getString("_UI_run_expr_type") + " " + label;
	}

	/**
	 * This handles model notifications by calling {@link #updateChildren} to update any cached
	 * children and by creating a viewer notification, which it passes to {@link #fireNotifyChanged}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void notifyChanged(Notification notification) {
		updateChildren(notification);

		switch (notification.getFeatureID(run_expr.class)) {
			case PromelaPackage.RUN_EXPR__PRIORITY:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), false, true));
				return;
			case PromelaPackage.RUN_EXPR__ARG_LST:
				fireNotifyChanged(new ViewerNotification(notification, notification.getNotifier(), true, false));
				return;
		}
		super.notifyChanged(notification);
	}

	/**
	 * This adds {@link org.eclipse.emf.edit.command.CommandParameter}s describing the children
	 * that can be created under this object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected void collectNewChildDescriptors(Collection<Object> newChildDescriptors, Object object) {
		super.collectNewChildDescriptors(newChildDescriptors, object);

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createvarref()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createbin_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createcond_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createlen_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createtimeout_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createnp__expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createenabled_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createpc_value_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createname_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createrun_expr()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createnum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createenum_const()));

		newChildDescriptors.add
			(createChildParameter
				(PromelaPackage.Literals.RUN_EXPR__ARG_LST,
				 PromelaFactory.eINSTANCE.createmtype_const()));
	}

	/**
	 * This returns the label text for {@link org.eclipse.emf.edit.command.CreateChildCommand}.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String getCreateChildText(Object owner, Object feature, Object child, Collection<?> selection) {
		Object childFeature = feature;
		Object childObject = child;

		boolean qualify =
			childFeature == PromelaPackage.Literals.STMNT__UNLESS ||
			childFeature == PromelaPackage.Literals.RUN_EXPR__ARG_LST;

		if (qualify) {
			return getString
				("_UI_CreateChild_text2",
				 new Object[] { getTypeText(childObject), getFeatureText(childFeature), getTypeText(owner) });
		}
		return super.getCreateChildText(owner, feature, child, selection);
	}

}
